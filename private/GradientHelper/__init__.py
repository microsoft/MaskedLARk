import logging
import gradcompute
import utils
import os

import azure.functions as func

import os 
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    is_body_size_over_limitation = utils.check_request_body_size(req)
    if(is_body_size_over_limitation != ''):
        return func.HttpResponse(
        is_body_size_over_limitation,
        status_code=400)

    helpername_source = 'helpername.txt'
    with open(helpername_source) as f:
        helpername = f.read()
        logging.info('Helper request name: ' + str(helpername))
    try:
        req = utils.check_and_convert_input(req, helpername)
        if req is None:
            raise ValueError('Helper parsing failed.')
        
        input_data, model_set, privacy_settings = utils.parse_adserver_json_payload(req)
        if input_data is None and model_set is None:
            raise ValueError('Helper parsing failed.')

        output_gradients = {}
        for model_name in model_set:
            model = model_set[model_name]['model']
            lossfunction = model_set[model_name]['loss']
            gradients = gradcompute.compute_private_gradients(input_data, model, model_name, lossfunction, privacy_settings)
            if gradients is None:
                # Typically this means not enough gradients were sent for a model to surpass the k-anonymity threshold
                output_gradients[model_name] = {}
            else:
                output_gradients[model_name] = gradients

        output_payload = utils.create_return_payload(output_gradients, helpername)
        return func.HttpResponse(output_payload)
    except TypeError:
        pass


    return func.HttpResponse(
            "Your request was not a valid request to the helper service.",
            status_code=400
    )
