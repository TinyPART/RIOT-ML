/*
 * Copyright (c) 2023 Zhaolan Huang
 * 
 * This file is subject to the terms and conditions of the GNU Lesser
 * General Public License v3.0. See the file LICENSE in the top level
 * directory for more details.
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "net/nanocoap.h"
#include "kernel_defines.h"
#include "mlmci.h"

extern mlmodel_t *model_ptr;

static ssize_t _model_status_handler(coap_pkt_t *pkt, uint8_t *buf, size_t len,
                                   coap_request_ctx_t *context)
{
    (void)context;
    mlmodel_status_t status = mlmodel_get_status(model_ptr);
    return coap_reply_simple(pkt, COAP_CODE_205, buf, len,
            COAP_FORMAT_TEXT, (uint8_t*)&status, 1);
}
NANOCOAP_RESOURCE(model_status) { 
    .path= "/model/status", .methods = COAP_GET, .handler = _model_status_handler
};

static ssize_t _model_name_handler(coap_pkt_t *pkt, uint8_t *buf, size_t len,
                                   coap_request_ctx_t *context)
{
    (void)context;
    const char* name = mlmodel_get_name(model_ptr);
    return coap_reply_simple(pkt, COAP_CODE_205, buf, len,
            COAP_FORMAT_TEXT, name, strlen(name));
}
NANOCOAP_RESOURCE(model_name) { 
    .path= "/model/name", .methods = COAP_GET, .handler = _model_name_handler
};

static ssize_t _model_eval_result_handler(coap_pkt_t *pkt, uint8_t *buf, size_t len,
                                   coap_request_ctx_t *context)
{
    (void)context;
    const char* name = mlmodel_get_name(model_ptr);
    return coap_reply_simple(pkt, COAP_CODE_205, buf, len,
            COAP_FORMAT_TEXT, name, strlen(name));
}
NANOCOAP_RESOURCE(model__eval_result) { 
    .path= "/model/eval_result", .methods = COAP_GET, .handler = _model_eval_result_handler
};

static ssize_t _model_stop_handler(coap_pkt_t *pkt, uint8_t *buf, size_t len,
                                   coap_request_ctx_t *context)
{
    (void)context;
    const char* name = mlmodel_get_name(model_ptr);
    return coap_reply_simple(pkt, COAP_CODE_205, buf, len,
            COAP_FORMAT_TEXT, name, strlen(name));
}
NANOCOAP_RESOURCE(model_stop) { 
    .path= "/model/stop", .methods = COAP_POST, .handler = _model_stop_handler
};

static ssize_t _model_run_handler(coap_pkt_t *pkt, uint8_t *buf, size_t len,
                                   coap_request_ctx_t *context)
{
    (void)context;
    const char* name = mlmodel_get_name(model_ptr);
    return coap_reply_simple(pkt, COAP_CODE_205, buf, len,
            COAP_FORMAT_TEXT, name, strlen(name));
}
NANOCOAP_RESOURCE(model_run) { 
    .path= "/model/run", .methods = COAP_POST, .handler = _model_run_handler
};

static ssize_t _model_run_eval_handler(coap_pkt_t *pkt, uint8_t *buf, size_t len,
                                   coap_request_ctx_t *context)
{
    (void)context;
    const char* name = mlmodel_get_name(model_ptr);
    return coap_reply_simple(pkt, COAP_CODE_205, buf, len,
            COAP_FORMAT_TEXT, name, strlen(name));
}
NANOCOAP_RESOURCE(model_run_eval) { 
    .path= "/model/run", .methods = COAP_POST, .handler = _model_run_eval_handler
};

static ssize_t _model_params_update_handler(coap_pkt_t *pkt, uint8_t *buf, size_t len,
                                   coap_request_ctx_t *context)
{
    (void)context;
    const char* name = mlmodel_get_name(model_ptr);
    return coap_reply_simple(pkt, COAP_CODE_205, buf, len,
            COAP_FORMAT_TEXT, name, strlen(name));
}
NANOCOAP_RESOURCE(model_params_update) { 
    .path= "/model/params/update", .methods = COAP_POST, .handler = _model_params_update_handler
};



