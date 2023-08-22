import argparse
import torch
from torchvision import transforms, models
import PIL
import json


def get_input_args():
    parser = argparse.ArgumentParser(
        description="Predict flower name from an image using a trained deep learning model."
    )
    parser.add_argument("input", type=str, help="Path to the image")
    parser.add_argument("checkpoint", type=str, help="Path to the model checkpoint")
    parser.add_argument("--top_k", type=int, default=5, help="Return to K predictions")
    parser.add_argument(
        "--category_names",
        type=str,
        default="cat_to_name.json",
        help="Path to the category to name mapping json",
    )
    parser.add_argument("--gpu", action="store_true", help="Use GPU for inference")
    return parser.parse_args()


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)

    if checkpoint["arch"] == "vgg13":
        model = models.vgg13(pretrained=True)
    elif checkpoint["arch"] == "vgg16":
        model = models.vgg16(pretrained=True)
    else:
        print(f"Architecture {checkpoint['arch']} not recognised.")
        return

    model.classifier = checkpoint["classifier"]
    model.load_state_dict(checkpoint["state_dict"])
    model.class_to_idx = checkpoint["class_to_idx"]
    return model


def process_image(image_path):
    image = PIL.Image.open(image_path)
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_tensor = transform(image)
    return image_tensor.unsqueeze_(0)


def predict(image_or_path, model, topk, device):
    if isinstance(image_or_path, str):
        image = process_image(image_or_path).to(device)
    else:
        # Convert PIL image to tensor and add match dimension
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image = transform(image_or_path).unsqueeze(0).to(device)

    model.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        probs, indices = outputs.topk(topk)
        probs = probs.exp().data.cpu().numpy()[0]
        indices = indices.data.cpu().numpy()[0]
        idx_to_class = {v: k for k, v in model.class_to_idx.items()}
        classes = [idx_to_class[idx] for idx in indices]
    return probs, classes


def main():
    args = get_input_args()
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    model = load_checkpoint(args.checkpoint)
    probs, classes = predict(args.input, model, args.top_k, device)

    with open(args.category_names, "r") as f:
        cat_to_name = json.load(f)

    class_names = [cat_to_name[cls] for cls in classes]

    print("Predictions for the image:", args.input)
    for p, cls in zip(probs, class_names):
        print(f"{cls}: {p:.3f}")

    print("\nPrediction completed successfully!")


if __name__ == "__main__":
    main()
