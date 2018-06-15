# [START setup]
# Import the Python client library.
import googleapiclient.discovery

# Define variables for your project ID, version name, and model name after
# you set them up on Cloud ML Engine.

PROJECT_ID = <YOUR PROJECT_ID HERE>
VERSION_NAME = <YOUR VERSION_NAME HERE>
MODEL_NAME = <YOUR MODEL_NAME HERE>
# [END setup]


# [START send-prediction-request]
service = googleapiclient.discovery.build('ml', 'v1')
name = 'projects/{}/models/{}'.format(PROJECT_ID, MODEL_NAME)
name += '/versions/{}'.format(VERSION_NAME)

# Due to the size of the data, it needs to be split in 2
first_half = test_features[:int(len(test_features)/2)]
second_half = test_features[int(len(test_features)/2):]

complete_results = []
for data in [first_half, second_half]:
    responses = service.projects().predict(
        name=name,
        body={'instances': data}
    ).execute()

    if 'error' in responses:
        print(response['error'])
    else:
        complete_results.extend(responses['predictions'])

# Print the first 10 responses
for i, response in enumerate(complete_results[:10]):
    print('Prediction: {}\tLabel: {}'.format(response, test_labels[i]))
# [END send-prediction-request]
