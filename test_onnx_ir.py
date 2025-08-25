import onnx

# # Load the original model
# model = onnx.load('ctdet_helmet.onnx')

# # Set the model IR version to 9
# model.ir_version = 11
# # Save the modified model
# onnx.save(model, 'ctdet_helmet_v11.onnx')


from onnx import version_converter

model = onnx.load("ctdet_helmet.onnx")
converted_model = version_converter.convert_version(model, target_version=9)  # Downgrade opset
onnx.save(converted_model, "ctdet_helmet_opset9.onnx")