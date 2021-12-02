import timm
m = timm.create_model('mixnet_xl', pretrained=True)
for key, val in m.state_dict().items():
    print(key)