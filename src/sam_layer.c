#include "sam_layer.h"
#include "utils.h"
#include "dark_cuda.h"
#include "blas.h"
#include <stdio.h>
#include <assert.h>

layer make_sam_layer(int batch, int index, int w, int h, int c, int w2, int h2, int c2)
{
    fprintf(stderr,"scale Layer: %d\n", index);
    layer l = { (LAYER_TYPE)0 };
    l.type = SAM;
    l.batch = batch;
    l.w = w;
    l.h = h;
    l.c = c;

    l.out_w = w2;
    l.out_h = h2;
    l.out_c = c2;
    assert(l.out_c == l.c);
    assert(l.w == l.out_w && l.h == l.out_h);

    l.outputs = l.out_w*l.out_h*l.out_c;
    l.inputs = l.outputs;
    l.index = index;

    l.delta = (float*)xcalloc(l.outputs * batch, sizeof(float));
    l.output = (float*)xcalloc(l.outputs * batch, sizeof(float));

    l.forward = forward_sam_layer;
    l.backward = backward_sam_layer;

    return l;
}

void resize_sam_layer(layer *l, int w, int h)
{
    l->out_w = w;
    l->out_h = h;
    l->outputs = l->out_w*l->out_h*l->out_c;
    l->inputs = l->outputs;
    l->delta = (float*)xrealloc(l->delta, l->outputs * l->batch * sizeof(float));
    l->output = (float*)xrealloc(l->output, l->outputs * l->batch * sizeof(float));



}

void forward_sam_layer(const layer l, network_state state)
{
    int size = l.batch * l.out_c * l.out_w * l.out_h;
    //int channel_size = 1;
    float *from_output = state.net.layers[l.index].output;

    int i;
    #pragma omp parallel for
    for (i = 0; i < size; ++i) {
        l.output[i] = state.input[i] * from_output[i];
    }

    activate_array(l.output, l.outputs*l.batch, l.activation);
}

void backward_sam_layer(const layer l, network_state state)
{
    gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);
    //axpy_cpu(l.outputs*l.batch, 1, l.delta, 1, state.delta, 1);
    //scale_cpu(l.batch, l.out_w, l.out_h, l.out_c, l.delta, l.w, l.h, l.c, state.net.layers[l.index].delta);

    int size = l.batch * l.out_c * l.out_w * l.out_h;
    //int channel_size = 1;
    float *from_output = state.net.layers[l.index].output;
    float *from_delta = state.net.layers[l.index].delta;

    int i;
    #pragma omp parallel for
    for (i = 0; i < size; ++i) {
        state.delta[i] += l.delta[i] * from_output[i]; // l.delta * from  (should be divided by channel_size?)

        from_delta[i] = state.input[i] * l.delta[i]; // input * l.delta
    }
}
