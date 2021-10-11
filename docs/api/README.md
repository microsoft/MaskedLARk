# Introduction

Welcome to the MaskedLARK Helper Service API documentation! This presents the documentation to make direct requests to the Aggregation and Gradient Helper services, along with details about their endpoint locations and returned values. For examples, please see: https://github.com/microsoft/MaskedLARk/src/RequestDemo

This API represents a work in progress.

# Global settings

### POST Request Body

The POST request has the following possible global settings -- these are needed for various computations that could be performed.

Parameter | Type | Required | Description
--------- | ------- | ------- | -----------
origin | string | yes | The name of the helper service that the request will be sent to.
function | string | yes | The desired functionality. For now, the only options are `aggregation`,`gradient_computation` and `evaluate`.
attributionreportto | string | yes | The reporting origin / adtech that's requesting the information. 
aggregation_service_payloads | List of dicts | yes | A list containing the data payloads. The data payload format is described below.
privacy_settings | dict | no| Privacy settings -- debug scenario for when PublicParameterServer is unavailable. The format is described below.

## Payload Set
The payload set describes the payloads that are sent as a list, combining data passed by various browsers.  Each payload has the following structure:

Parameter | Type | Required | Description
--------- | ------- | ------- | -----------
encryption_standard | string | yes | How to decrypt this.  Only accepts `cleartext` for now
function | string | no | The function that will be using this payload.  This will be declared by the browser and can be stored / passed on if desired, but the higher-level `function` above will override this regardless.
attributionreportto | string | yes | Must match the `attributionreportto` key in the higher level POST.  This is to ensure adtech uses their publicly declared privacy settings, and not another origin
payload | string | yes | The payload -- each key value pair provides data for the corresponding functionality

Encryption is not supported at this time. **Please do not use sensitive data**, and send unencrypted payloads until encryption is properly supported. For now, the payload must be a cleartext dict.


The highest level of a JSON body of a request that might be sent to the aggregation function:

```json
{
    "origin" : "<helper service>",
    "function" : "aggregation",
    "attributionreportto" : "adtech.com",
    "aggregation_service_payloads" : [
        {
            "encryption_standard" : "cleartext",
            "function" : "aggregation",
            "attributionreportto" : "adtech.com",
            "payload" : {
                aggregation_key: {"campaign" : 100, "location" : "seattle"}
                aggregation_values: {"view" : 1, "impression" : 1, "click" : 1, "purchase" : 123}
            }
        },
        ...
    ]
}
```


## Privacy Settings
In practice, the privacy settings per `attributionreportto` will be publicly declared on the Parameter Server.  For the purposes of debugging, we allow testers to override those settings when a) the `privacy_settings` are declared and b) all payloads encryption is set to `cleartext`.

Below are the options for the `privacy_settings` field, if declared:

Parameter | Type | Required | Description
--------- | ------- | ------- | -----------
version | string | yes | The version of privacy settings
conversion_range | dict | yes| A dictionary mapping an English tag to a bound id, e.g., {"purchase" : [0, 1024] }.  If the bound value is a two item list, it defines a real valued range, if it's a set it's a categorical set.
<attributionreportto> | dict | yes | A dictionary setting mapping the version id to the settings

Below are the options for the `<attributionreportto>`[version] field, found in the `privacy_setttings`, if declared:

Parameter | Type | Required | Description
--------- | ------- | ------- | -----------
"aggregation" | dict | yes | Sets the privacy settings for the aggregation service
"gradient_computation" | dict | yes | Sets the privacy settings for the gradient computation service

A sample of the `privacy_settings` field would look like:
```json
{
    "origin" : "adtech.org",
    "function" : "aggregation",
    "aggregation_service_payloads" : [ ... ],
    "privacy_settings" : {
        "version" : "V0.01",
        "conversion_range" : {"purchase" : [0, 1024] },
        "<attributionreportto>" : {
            "V0.01" : {
                "aggregation" : {
                    "k-anonymity" : {"mechanism" : "standard", "threshold" : 50, }
                    , "differential_privacy": {"mechanism" : "laplace", "epsilon" : 1}
                    ...
                },
                "gradient_computation" : {
                    "k-anonymity" : {"mechanism" : "standard", "threshold" : 50 }
                    , "differential_privacy": {"mechanism" : "laplace", "epsilon" : 1, "gradient_bound" : 1}
                    ...
                },
                ... 
            },
            "V0.02" : {
                "aggregation" : {
                    "k-anonymity" : {"mechanism" : ..., "threshold" : ..., }
                    , "differential_privacy": {"mechanism" : ..., "epsilon" : ..., }
                    ...
                },
                "gradient_computation" : {
                    "k-anonymity" : {"mechanism" : ..., "threshold" : ..., }
                    , "differential_privacy": {"mechanism" : ..., "epsilon" : ..., "norm_bound" : ...}
                    ...
                },
                ... 
            },
            ...
        }
    }
}
```

# Aggregation Service Helper

The aggregation service has certain parameters only available to it.  These specify the queries or groupby that should be performed.


Parameter | Type | Required | Description
--------- | ------- | ------- | -----------
origin | string | yes | The name of the helper service that the request will be sent to.
function | string | yes | Set to `aggregation`
aggregation_service_payload_set | List of dicts | yes | A list containing the data payloads. The data payload format is described below.
aggregation_service_queries | List of dicts | no | A list of JSON specifying individual queries that should be executed.
aggregation_service_groupby | List of list of strings | no | A list of lists, where the inner lists specify aggregation keys to be grouped over.
privacy_settings | dict | no| Privacy settings -- debug scenario for when PublicParameterServer is unavailable. The privacy setting fields are described above.


 A sample POST request for specific queries might be the following:
```json
{
    "origin": "servicehelpergradients1",
    "function" : "aggregation",
    "privacy_settings" : { },
    "aggregation_service_payload_set" : [ ... ],
    "aggregation_service_queries" : [
        {"location" : "seattle", "campaign: "100"}
        , {"location" : "new york", "campaign: "100"}
        ...
        , {"location" : "seattle", "campaign: "101"}
    ],
    "aggregation_service_groupby" : [
        ["location"]
        , ["location", "campaign"]
        ...
        , ["location", "campaign", "language"]
    ]
}
```

## Aggregation Payload Set
For the `payload_set`, the aggregation service utilizes the following:

Parameter | Type | Required | Description
--------- | ------- | ------- | -----------
aggregation_key | dict | yes | Contextual features to be aggregated
aggregation_values | dict | yes | Key-value pairs for different types of conversions.  The type must be declared by the `conversion_range` above, and the value must lie within the bound.

> An example JSON for the aggregation service's payload

```json
"payload" : {
    aggregation_key: {"campaign" : 100, "location" : "seattle"}
    aggregation_values: {"view" : 1, "impression" : 1, "click" : 1, "purchase" : 123}
}
```

## Aggregation Privacy Settings

The aggregation service has declared privacy settings with the following options:

Parameter | Type | Required | Description
--------- | ------- | ------- | -----------
k-anonymity | dict | yes | Settings for k-anonymity `{"mechanism" : "standard", "threshold" : 50, }`
differential_privacy| dict | yes | Settings for differential privacy `{"mechanism" : "laplace", "epsilon" : 1}`

# Gradient Service Helper

The gradient service helper is designed to be used to train small neural network models in a private fashion. 

The gradient service helper has limitations on the types of models and losses that can be used at this time. In particular, only feedforward networks with standard activation functions (ReLU, Leaky ReLU, Sigmoid, ELU, tanh), with less than 200 dimensions per layer are permitted. It is strongly suggested to limit the network to far less than that limit, and to limit the depth of the network for communications and latency reasons.

The types of models that the helpers expect are ONNX model representations. The MaskedLARK PyPi package can be used at this time to seamlessly train PyTorch models, and bindings for Tensorflow and other frameworks are a work in progress. That same package can be used to generate correctly-formatted payloads as defined below, but this package is not required to use the gradient helpers.

## Retrieve Gradients

Retrieving gradients is done through a POST request to a gradient helper endpoint.

### POST Request Body

The POST request must contain all of the following parameters.

Parameter | Type | Required | Description
--------- | ------- | ------- | -----------
origin | string | yes | The name of the helper service that the request will be sent to.
function | string | yes | The desired functionality. "gradient_computation" triggers this API
gradient_service_payload_set | List of dicts | yes | A list containing the data payloads. The data payload format is described below.
gradient_model_set | List of dicts | yes | A list containing the model and loss payloads. The model payload format is described below.
privacy_settings | dict | no | Privacy settings -- debug scenario for when PublicParameterServer is unavailable. The privacy setting fields are described above.

### Privacy Settings

Parameter | Type | Required| Description
--------- | ------- | ------ | -----------
k-anonymity | dict | yes | Settings for k-anonymity thresholds
differential_privacy | dict | yes | Settings for differential privacy noise


*K-anonymity Settings*
Parameter | Type | Required| Description
--------- | ------- | ------ | -----------
mechanism | string | yes | Currently only `standard` is supported
threshold| string | yes | For a gradient to be returned, we need a batch of at least this size


*Differential Privacy Settings*
Parameter | Type | Required | Description
--------- | ------- | ------- | -----------
method | string, default "laplacian" | yes | The type of noise to be used when producing the gradients. For now, the only option available is "laplacian".
epsilon | float, default 1.0 | no | A noise parameter that corresponds to the differential privacy guarantee. Alternately, the scale parameter of the noise.
norm_bound | float, default 1.0 | no | The value used when clipping gradients before aggregation. All gradients with L2 norm greater than norm_bound will be rescaled to have L2 norm of norm_bound.

As described in the documentation, the gradient helpers will ensure differentially-private gradients by first clipping gradients that are too large, then by adding noise with magnitude epsilon.

An example JSON of privacy settings for the gradient helpers:
```json
"gradient_computation" : {
    "k-anonymity" : {"mechanism" : "standard", "threshold" : 50 }
    , "differential_privacy": {"mechanism" : "laplace", "epsilon" : 1, "gradient_bound" : 1}
    ...
},
```

### Data payload format

The data payloads contain a dictionary with the following fields:

Parameter | Type | Required | Description
--------- | ------- | ------- | -----------
encryption | string | yes | The type of encryption used on the payloads. For now, this field is not used, as encryption is not implemented.
payload | dict | yes | The payload containing a data point and one or more targets to compute gradients against.


The payload format is as follows:

Parameter | Type | Required | Description
--------- | ------- | ------- | -----------
model_features | list of ints | yes | The model features used as input to the model. The model features must be a single list of integers. Note that two-dimensional lists of lists are not supported.
model_labels | list of ints | yes | The targets used to compute gradients. Multiple targets can (and should) be sent; the results will be aggregated after gradient computation. 
model_masks | list of ints | yes | Masks used for gradient privacy. This list must be the same length as "model_labels".
model_tag | string OR list of strings | yes | The name of the model(s) that model_features should be fed to.

Please note that the inputs to the networks can only be sent as ints. The gradient helper will convert the sent ints into floats before using them as inputs to the model, so it is recommended to rescale the inputs using the first layer of your feedforward network.

### Model payload format

The model payload is also structured. The fields are outlined below:

Parameter | Type | Required | Description
--------- | ------- | ------- | -----------
model_tag | string | yes | A unique identifier
model_loss_function | dict | yes | A description of the loss function to use to calculate gradients. The format is described below.
model | string | yes | The string resulting from calling onnx._serialize on an ONNX model describing the model being used. The stream must be encoded as a string in the latin1 format.

To emphasize: the gradient helper must receive the model in a string version of a serialized ONNX model, encoded in the latin1 format.

A sample POST request might be the following:

```json
[
    {
        "origin": "servicehelpergradients1",
        "function" : "gradient_computation",
        "privacy_settings" : {...},
        "gradient_service_payload_set" : [
            {
                "gradient_service_payload" : {
                    "encryption_standard" : "cleartext",
                    "payload" : [
                        {
                            "model_features": [23,0,2,89,205],
                            "model_labels" : [0, 1],
                            "model_masks" : [-4, 3],
                            "model_tag" : "ConversionModel11023"
                        },
                        {
                            "model_features" : [14,128,3,97,200], 
                            "model_labels": [0, 1],
                            "model_masks": [-8, -16],
                            "model_tag": ["ConversionModel110018"]
                        }
                        ... 
                    ]
                }
            },
        ],
        "gradient_model_set" : [
            {
                "model_tag" : "ConversionModel11023",
                "model_loss_function" : {
                    "loss_type" : "singleton",
                    "loss_name" : "BCELoss",
                    "loss_kwargs" : {}
                },
                "model" : model0_proto_string
            },
            {
                "model_tag" : "ConversionModel110018",
                "model_loss_function" : {
                    "loss_type" : "singleton",
                    "loss_name" : "MSELoss",
                    "loss_kwargs" : {"weight" : [0.7,0.3]}
                },
                "model" : model1_proto_string
            }
        ]
    }
]
```

### Returned Data

In order to return the gradients with an associated shape, the gradients are returned so that they can be simply reconstructed into numpy arrays. This format is as a dictionary with the following fields:

Parameter | Type | Description
--------- | ------- | -----------
\_\_ndarray\_\_ | list (of lists [of lists etc.]) of floats | A list of floats containing the values of the numpy array. If the numpy array was two-dimensional, this should be a list of lists. If 3, a list of lists of lists, and so forth. This is the output of .tolist() on a numpy array.
dtype | string | A valid numpy datatype, i.e. float32, float64, int32.
shape | list of ints | The shape of the original numpy array, to ensure proper reconstruction.

The JSON that is returned from a proper request is structured like this:

```json
[
  {"origin": "servicehelpergradients1", 
  "model_gradient_set": 
    [
      {
      "model_tag": "ConversionModel110021", 
      "model_noisy_gradients": {
        "Gemm_7.weight": 
          {
            "__ndarray__": 
            [[-0.01077650859951973, -0.11328306049108505, -0.04680103808641434, 0.12764239311218262, -0.04177387058734894, 0.011603852733969688, 0.20162659883499146, -0.01936546340584755, 0.022513577714562416, -0.006226115860044956], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [-0.06162182241678238, -0.6477709412574768, -0.26761594414711, 0.7298798561096191, -0.2388697862625122, 0.0663527175784111, 1.1529335975646973, -0.11073485761880875, 0.12873628735542297, -0.035601940006017685], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
            "dtype": "float32", 
            "shape": [1, 80]
          }, 
        ... 
        "Gemm_11.bias": 
          {
            "__ndarray__": [2.7372794151306152], 
            "dtype": "float32", 
            "shape": [1]
          }
      }
    ]
  }
]
```

## Evaluate Models

Retrieving model evaluations is done through a POST request to a helper endpoint. The helper endpoint API is identical to the gradient helper service, with the difference that the "function" field must be set to "evaluation". The norm bound used for gradient computation is not used during evaluation.



An evaluation request returns JSON structured like this:

```json
{
  "origin": "servicehelpergradients1",
  "model_evaluation_result": [
    {
      "model_tag" : "ConversionModel110110",
      "model_loss" : 0.0021
    },
    ...
    {
      "model_tag" : "ConversionModel110115",
      "model_loss" : 1.2110
    }
  ]
}
```

