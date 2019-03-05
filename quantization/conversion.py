#

if __name__ == "__main__":
    import coremltools
    coreml_model = coremltools.converters.caffe.convert(('oxford102.caffemodel', 'deploy.prototxt'), image_input_names='data', class_labels='class_labels.txt')
    coreml_model.save('oxford102.mlmodel')

    # reduced precision model
    model_spec = coremltools.utils.load_spec('./oxford102.mlmodel')
    model_fp16_spec = coremltools.utils.convert_neural_network_spec_weights_to_fp16(model_spec)
    coremltools.utils.save_spec(model_fp16_spec, 'oxford102_fp16.mlmodel')



