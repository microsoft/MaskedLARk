import logging

import azure.functions as func

from BaseHelper import utils, losscompute

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    is_body_size_over_limitation = utils.check_request_body_size(req)
    if(len(is_body_size_over_limitation) > 0):
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

        output_losses = {}
        for model_name in model_set:
            model = model_set[model_name]['model']
            lossfunction = model_set[model_name]['loss']
            losses = losscompute.compute_private_losses(input_data, model, model_name, lossfunction, privacy_settings)
            if losses is None:
                output_losses[model_name] = {}
            else:
                output_losses[model_name] = losses

        output_payload = utils.create_eval_return_payload(output_losses, helpername)
        return func.HttpResponse(output_payload)
    except ValueError:
        pass


    return func.HttpResponse(
            "Your request was not a valid request to the helper service.",
            status_code=400
    )
