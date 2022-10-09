import segmentation_models as sm

def model_compile():
  best_params = {
      'alpha':0.007231,
      'gamma':1.923258,
      'epsilon':0.595344
  }
  sm.set_framework('tf.keras')
  BACKBONE = 'resnet34'
  preprocess_input = sm.get_preprocessing(BACKBONE)


  # define model
  model = sm.Unet(BACKBONE, encoder_weights='imagenet')
  model.compile(
      'Adam',
      loss=sm.losses.BinaryFocalLoss(alpha=best_params['alpha'], gamma=best_params['gamma']),
      metrics=[sm.metrics.precision],
  )
  return model