import torch, torchvision

from PIL import Image
from torchvision import transforms
input_image = Image.open('Apple.jpg')
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# Read the categories
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

alexnet_model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
alexnet_model.eval()

resnet_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)
resnet_model.eval()



with torch.no_grad():
    output_alexnet = alexnet_model(input_batch)
alexnet_probabilities = torch.nn.functional.softmax(output_alexnet[0], dim=0)


with torch.no_grad():
    output_resnet = resnet_model(input_batch)
resnet_probabilities = torch.nn.functional.softmax(output_resnet[0], dim=0)


# Show top categories per image
top5_prob, top5_catid = torch.topk(alexnet_probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())

# Show top categories per image
top5_prob, top5_catid = torch.topk(resnet_probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())
