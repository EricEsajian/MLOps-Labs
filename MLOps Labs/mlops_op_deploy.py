import boto3
import io
import zipfile
import json
import logging
from crhelper import CfnResource
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)
# Initialise the helper, all inputs are optional, this example shows the defaults
helper = CfnResource(json_logging=False, log_level='DEBUG', boto_level='CRITICAL')

s3 = boto3.client('s3')
sm =  boto3.client('sagemaker')
auto = boto3.client('application-autoscaling')
cfn = boto3.client('cloudformation')

def lambda_handler(event, context):
    helper(event, context)

def create_autoscaling_policy(event, context):
    endpoint_name = helper.Data.get('endpoint_name')
    variant_names = helper.Data.get('variant_names')
    meta = helper.Data.get('deployment_metadata')
    role_arn = helper.Data.get('role_arn')
    for variant_name in variant_names:
        resourceId='endpoint/%s/variant/%s' % (endpoint_name, variant_name)
        resp = auto.register_scalable_target(
            ServiceNamespace='sagemaker',
            ResourceId=resourceId,
            ScalableDimension='sagemaker:variant:DesiredInstanceCount',
            MinCapacity=meta['AutoScaling']['MinCapacity'],
            MaxCapacity=meta['AutoScaling']['MaxCapacity'],
            RoleARN=role_arn
        )
        resp = auto.put_scaling_policy(
            PolicyName='%s-%s' % (endpoint_name, variant_name),
            PolicyType='TargetTrackingScaling',
            ResourceId=resourceId,
            ScalableDimension='sagemaker:variant:DesiredInstanceCount',
            ServiceNamespace='sagemaker',
            TargetTrackingScalingPolicyConfiguration={
                'TargetValue': meta['AutoScaling']['TargetValue'],
                'ScaleInCooldown': meta['AutoScaling']['ScaleInCooldown'],
                'ScaleOutCooldown': meta['AutoScaling']['ScaleOutCooldown'],
                'PredefinedMetricSpecification': {'PredefinedMetricType': meta['AutoScaling']['PredefinedMetricType'] }
            }
        )

def prepare_descriptors(event, context):
    deployment_info = None
    env = event['ResourceProperties']['Environment']
    job_name = event['ResourceProperties']['JobName']
    job_description = sm.describe_training_job(TrainingJobName=job_name)
    
    if not env in ["development", "production"]:
        raise Exception( "Invalid deployment environment: %s" % env)
    
    resp = s3.get_object(Bucket=event['ResourceProperties']['AssetsBucket'], Key=event['ResourceProperties']['AssetsKey'])
    with zipfile.ZipFile(io.BytesIO(resp['Body'].read()), "r") as z:
        deployment_info = json.loads(z.read('deployment.json').decode('ascii'))
    
    metadata = deployment_info['DevelopmentEndpoint'] if env == 'development' else deployment_info['ProductionEndpoint']

    # Now create the Endpoint Configuration
    endpoint_name = "%s-%s" % (deployment_info['EndpointPrefix'], env)
    endpoint_config_name = "%s-ec-%s-%s" % (deployment_info['EndpointPrefix'], job_name, env)
    endpoint_config_params = { 'EndpointConfigName': endpoint_config_name }
    endpoint_params = { 'EndpointName': endpoint_name, 'EndpointConfigName': endpoint_config_name }
    
    endpoint_config_params['ProductionVariants'] = [{
        'VariantName': 'model-a',
        'ModelName': job_name,
        'InitialInstanceCount': metadata["InitialInstanceCount"],
        'InstanceType': metadata["InstanceType"]
    }]
    variant_names=['model-a']
    # here we check if there is already a variant in the endpoint
    # we need to rearange the varia nts to do A/B tests
    if metadata['ABTests']:
        try: 
            resp = sm.describe_endpoint( EndpointName=endpoint_name)
            logger.info("Endpoint config name: %s", resp['EndpointConfigName'])
            resp = sm.describe_endpoint_config( EndpointConfigName=resp['EndpointConfigName'])
            old_variant = resp['ProductionVariants'][0]
            old_variant['InitialVariantWeight'] = 1.0 - metadata['InitialVariantWeight']
            
            new_variant = endpoint_config_params['ProductionVariants'][0]
            new_variant['VariantName'] = "model-b" if old_variant['VariantName'].endswith('-a') else "model-a"
            new_variant['InitialVariantWeight'] = metadata['InitialVariantWeight']
            
            endpoint_config_params['ProductionVariants'].append(old_variant)
            variant_names.append('model-b')
        except Exception as ex:
            logger.info("Error while trying to retrieve the EndpointConfig. It means we'll have just one variant: %s", ex)
            
    # here we enable the log writing if required
    if metadata['InferenceMonitoring']:
        endpoint_config_params['DataCaptureConfig'] = {
            'EnableCapture': True,
            'InitialSamplingPercentage': metadata['InferenceMonitoringSampling'],
            'DestinationS3Uri': metadata['InferenceMonitoringOutputBucket'],
            'CaptureOptions': [{'CaptureMode': 'Input'},{'CaptureMode': 'Output'}],
            'CaptureContentTypeHeader': {
                "CsvContentTypes": ["text/csv"], "JsonContentTypes": ["application/json"]
            }
        }
    # now, we create the model metadata
    model_params = {
        'ModelName': job_name,
        'PrimaryContainer': {
            'Image': job_description['AlgorithmSpecification']['TrainingImage'],
            'ModelDataUrl': job_description['ModelArtifacts']['S3ModelArtifacts'],
        },
        'ExecutionRoleArn': job_description['RoleArn']
    }
    helper.Data.update({
        'endpoint_name': endpoint_name, 
        'endpoint_config_name': endpoint_config_name,
        'model_name': job_name,
        'deployment_metadata': metadata,
        'variant_names': variant_names,
        'role_arn': job_description['RoleArn'],
        'enable_auto_scaling': True if metadata['AutoScaling'] else False
    })
    return model_params, endpoint_config_params, endpoint_params

@helper.create
@helper.update
def start_deployment(event, context):
    
    try:
        model_params, endpoint_config_params, endpoint_params = prepare_descriptors(event, context)
        model_name = helper.Data.get('model_name')
        endpoint_name = helper.Data.get('endpoint_name')
        endpoint_config_name = helper.Data.get('endpoint_config_name')
        # and here we create all the three elements for a deploy
        try:
            sm.describe_model(ModelName=model_name)
        except ClientError as e:
            if e.response['Error']['Code'] == 'ValidationException':
                sm.create_model(**model_params)
        sm.create_endpoint_config(**endpoint_config_params)
        try:
            sm.describe_endpoint(EndpointName=endpoint_name)
            sm.update_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_config_name,
                RetainAllVariantProperties=False
            )
            logger.info("Endpoint found. Updating: %s", endpoint_name)
        except ClientError as e:
            if e.response['Error']['Code'] == 'ValidationException':
                sm.create_endpoint(**endpoint_params)
                logger.info("Endpoint wasn't found. Creating: %s", endpoint_name)
    except Exception as e:
        logger.error("start_deployment - Ops! Something went wrong: %s" % e)
        raise e


@helper.poll_create
@helper.poll_update
def check_deployment_progress(event, context):
    answer = False
    try:
        endpoint_name = helper.Data.get('endpoint_name')
        enable_auto_scaling = helper.Data.get('enable_auto_scaling')
        resp = sm.describe_endpoint(EndpointName=endpoint_name)
        status = resp['EndpointStatus']
        if status in ['Creating', 'Updating']:
            logger.info( "check_deployment_progress - Preparing endpoint %s, status: %s", endpoint_name, status)
        elif status == 'InService':
            logger.info( "check_deployment_progress - Endpoint %s ready to be used!", endpoint_name)
            if enable_auto_scaling:
                create_autoscaling_policy(event, context)
            answer = True
        else:
            answer = True
            raise Exception("Invalid state for endpoint %s: %s",  endpoint_name, resp['FailureReason'])
    except Exception as e:
        logger.error("check_deployment_progress - Ops! Something went wrong: %s" % e)
        if answer:
            raise e
    return answer

@helper.delete
def delete_deployment(event, context):
    try:
        env = event['ResourceProperties']['Environment']
        job_name = event['ResourceProperties']['JobName']
        logical_id = event['LogicalResourceId']
        request_physical_id = event['PhysicalResourceId']
        
        if not env in ["development", "production"]:
            raise Exception( "Invalid deployment environment: %s" % env)
            
        stack_name = event['StackId'].split('/')[1]
        resp = cfn.describe_stack_resource(StackName=stack_name, LogicalResourceId=logical_id)
        current_physical_id = resp['StackResourceDetail']['PhysicalResourceId']
        
        if request_physical_id != current_physical_id:
            logger.info("delete_deployment - Delete request for resouce id: %s, but the current id is: %s. Ignoring...", 
                request_physical_id, current_physical_id)
            helper.Data.update({'delete_old_resource': True})
            return
        
        resp = s3.get_object(Bucket=event['ResourceProperties']['AssetsBucket'], Key=event['ResourceProperties']['AssetsKey'])
        with zipfile.ZipFile(io.BytesIO(resp['Body'].read()), "r") as z:
            deployment_info = json.loads(z.read('deployment.json').decode('ascii'))
        endpoint_name = "%s-%s" % (deployment_info['EndpointPrefix'], env)
        endpoint_config_name = "%s-ec-%s-%s" % (deployment_info['EndpointPrefix'], job_name, env)
    
        helper.Data.update({'endpoint_name': endpoint_name, 'endpoint_config_name': endpoint_config_name})
        resp = sm.describe_endpoint(EndpointName=endpoint_name)
        status = resp['EndpointStatus']
        if status != 'InService':
            raise Exception( "You can't delete an endpoint that is not InService: %s, status[%s]" % (endpoint_name, status))
        else:
            sm.delete_endpoint(EndpointName=endpoint_name)
    except ClientError as e:
        if e.response['Error']['Code'] == 'ValidationException':
            logger.info("Well, there is no endpoint to delete: %s - %s", endpoint_name, e)
    except Exception as e:
        logger.error("delete_deployment - Ops! Something went wrong: %s" % e)
        raise e

@helper.poll_delete
def check_delete_deployment_progress(event, context):
    endpoint_name = None
    endpoint_config_name = None
    try:
        delete_old_resource = helper.Data.get('delete_old_resource')
        if delete_old_resource:
            logger.info("check_delete_deployment_progress - Nothing to do... ignoring")
            return True
            
        endpoint_name = helper.Data.get('endpoint_name')
        endpoint_config_name = helper.Data.get('endpoint_config_name')
        
        
        resp = sm.describe_endpoint(EndpointName=endpoint_name)
        status = resp['EndpointStatus']
        if status != 'Deleting':
            raise Exception('Error while trying to delete the endpoint: %s. Status: %s' % (endpoint_name, status) )
        else:
            return False
    except ClientError as e:
        if e.response['Error']['Code'] == 'ValidationException':
            logger.info("Finished! We deleted the endpoint. Now, let's delete the ec")
            try:
                sm.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
            except ClientError as ex:
                if e.response['Error']['Code'] == 'ValidationException':
                    logger.info("The EC wasn't created before, so let's ignore")
                else:
                    raise ex
    except Exception as e:
        logger.error("check_delete_deployment_progress - Ops! Something went wrong: %s" % e)
        raise e
    return True


