import json
import requests
from pprint import pprint


BASEURL=' https://mvhmezhcya.execute-api.us-west-2.amazonaws.com/prod/'

USER1 = 'wbnBPXqvip1XJZH0L7AFQ6iWm203Ijmg8sdhKTqk'

if __name__ == '__main__':

        HEADER = 'x-api-key'
        fqdn_endpoint = BASEURL + '/predict'
        fqdn_api_key = USER1
        right = float(0)
        wrong = float(0)
        with open('assignment.csv', mode='r', encoding='utf-8') as ih:
            line = ih.readline()
            while True:
                line = ih.readline()
                if line == '':
                    break
                tokens = line.split(',')
                domain = tokens[0]
                threat = tokens[1]
                domain = domain.lstrip().rstrip().lower()
                threat = threat.lstrip().rstrip().lower()
                fqdn = domain
                payload = {'fqdn': fqdn }
                headers = {HEADER: fqdn_api_key}
                response = requests.get(url=fqdn_endpoint, headers=headers, params=payload)
                if response.reason != 'OK':
                    print('FAIL')
                    pprint(json.loads(response.text))
                blob = json.loads(response.text)
                if blob['fqdn'] != fqdn:
                    print('FAIL')
                if (blob['dga'] == True and threat == 'dga') or (blob['dga'] == False and threat == 'benign'):
                    right += 1.0
                else:
                    wrong += 1.0

        print('Right: ', right, ' Wrong: ', wrong, 'Score: ', (right / (right + wrong)) * 100.0)
