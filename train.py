import argparse
import torch
from torchvision import datasets, transforms, models
import torch.nn as nn
import torch.optim as optim


def get_input_args():
    parser = argparse.ArgumentParser(
        description="Train a new network on a dataset and save the model as a checkpoint."
    )
    parser.add_argument("data_directory", type=str, help="Path to thedata directory")
    parser.add_argument(
        "--save_dir", type=str, default=".", help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="vgg13",
        choices=["vgg13", "vgg16"],
        help="Model architecture (vgg13 or vgg16)",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.01, help="Learning rate"
    )
    parser.add_argument(
        "--hidden_units", type=int, default=512, help="Number of hidden units"
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training")
    return parser.parse_args()


def main():
    args = get_input_args()
    print("Script started with arguments:", args)
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    # Define transforms for training and validation sets
    train_transforms = transforms.Compose(
        [
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    valid_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    print("Loading datasets...")
    # Load datasets with ImageFolder
    train_data = datasets.ImageFolder(
        args.data_directory + "/train", transform=train_transforms
    )
    valid_data = datasets.ImageFolder(
        args.data_directory + "/valid", transform=valid_transforms
    )
    print("Datasets loaded successfully!")

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)

    print(f"Building and training model with architecture: {args.arch}...")
    # Build and train the network
    if args.arch == "vgg13":
        model = models.vgg13(pretrained=True)
    elif args.arch == "vgg16":
        model = models.vgg16(pretrained=True)
    print("Model built successfully!")

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(
        nn.Linear(25088, args.hidden_units),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(args.hidden_units, len(train_data.find_classes)),
        nn.LogSoftmax(dim=1),
    )

    model.classifier = classifier

    model.class_to_idx = train_data.class_to_idx

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    model.to(device)

    # Training loop
    print(f"Training for {args.epochs} epochs...")
    epochs = args.epochs
    steps = 0
    print_every = 5

    for epoch in range(epochs):
        running_loss = 0
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation loop
        valid_loss = 0
        accuracy = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in validloader:
                inputs, labels = inputs.to(device), labels.to(device)
                logps = model.forward(inputs)
                batch_loss = criterion(logps, labels)
                valid_loss += batch_loss.item()

                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        print(
            f"Epoch {epoch+1}/{epochs}.."
            f"Train loss: {running_loss/print_every:.3f}.."
            f"Validation loss: {valid_loss/len(validloader):.3f}.."
            f"Validation accuracy: {accuracy/len(validloader):.3f}"
        )
        running_loss = 0
        model.train()

    print("Training completed successfully!")

    # Save the checkpoint
    print("Saving checkpoint...")
    checkpoint = {
        "arch": args.arch,
        "classifier": model.classifier,
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "class_to_idx": model.class_to_idx,
        "epochs": args.epochs,
    }

    torch.save(checkpoint, args.save_dir + "/checkpoint.pth")
    print(f"Checkpoint saved successfully at {args.save_dir}/checkpoint.pth!")


if __name__ == "__main__":
    main()
