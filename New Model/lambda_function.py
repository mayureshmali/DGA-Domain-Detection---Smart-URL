import json
import logging
import boto3
import time
import tldextract

from pprint import pprint

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def predict_one_dga_value(sm_client, features, endpoint_name):
    # print('Using model endpoint {} to predict dga for this feature vector: {}'.format(endpoint_name, features))
    is_dga = False
    body = features + '\n'
    start_time = time.time()

    response = sm_client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='text/csv',
        Body=body)
    predicted_value = json.loads(response['Body'].read())
    duration = time.time() - start_time
    if predicted_value > 0.5:
        is_dga = True
    return is_dga


VALID_CHARS = 'abcdefghijklmnopqrstuvwxyz0123456789-_'
LOOKUP_TABLE = None


def encode_fqdn(fqdn='www.google.com'):
    global VALID_CHARS
    global LOOKUP_TABLE
    if not LOOKUP_TABLE:
        LOOKUP_TABLE = dict()
        idx = 1
        for c in VALID_CHARS:
            LOOKUP_TABLE[c] = int(idx)
            idx += int(1)

    ds = tldextract.extract(fqdn)
    domain = ds.domain
    rvalue = list()
    for c in domain:
        rvalue.append(str(LOOKUP_TABLE[c]))
    for _ in range(len(rvalue), 63):
        rvalue.append('0')
    return ','.join(rvalue)


def lambda_handler(event, context):
    runtime_sm_client = boto3.client(service_name='sagemaker-runtime')
    if 'queryStringParameters' not in event:
        return {
            'statusCode': 227,
            'body': json.dumps('MISSING queryStringParameters')
        }
    else:
        query_parms = event['queryStringParameters']
        logger.info(msg=query_parms)
        features = encode_fqdn(fqdn=query_parms['fqdn'])
        p = predict_one_dga_value(sm_client=runtime_sm_client, features=features, endpoint_name='dga-endpoint-0')
        query_parms['dga'] = p
        logger.info(msg=query_parms)
        return {
            'statusCode': 200,
            'body': json.dumps(query_parms)
        }

