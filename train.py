# import torch
# from simulator.game15 import *
# from util.enums import *
# from util.utils import preprocess_state, generate_dataset, train_supervised_cnn
# from cnn_model import CNNModel
# import matplotlib.pyplot as plt

# def run_tests(model, env, num_tests=100, max_shuffle=10, step_limit=100):
#     successful_tests = 0

#     for _ in range(num_tests):
#         env.reset()
#         env.shuffle_n(max_shuffle)

#         state = env.get_state()
#         steps = 0
#         while not env.is_solved() and steps < step_limit:
#             steps += 1

#             state_tensor = preprocess_state(state, model.grid_size).to(model.device)

#             action_logits = model(state_tensor)
#             action = torch.argmax(action_logits).item() + 1

#             env.process_action(action)
#             state = env.get_state()

#         if env.is_solved():
#             successful_tests += 1

#     return successful_tests / num_tests

# def train_and_plot(model, dataset, epochs, learning_rate=0.001):
#     criterion = torch.nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#     losses = []
#     train_accuracies = []
#     test_accuracies = []

#     for epoch in range(epochs):
#         total_loss = 0
#         correct_train = 0

#         for state, action in dataset:
#             state_tensor = preprocess_state(state, model.grid_size).to(model.device)
#             target_action = torch.tensor([action.value - 1], dtype=torch.long).to(model.device)

#             action_logits = model(state_tensor)
#             loss = criterion(action_logits, target_action)

#             if torch.argmax(action_logits) == target_action:
#                 correct_train += 1

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()

#         avg_loss = total_loss / len(dataset)
#         train_accuracy = correct_train / len(dataset)

#         losses.append(avg_loss)
#         train_accuracies.append(train_accuracy)
#         test_accuracies.append(run_tests(model, Grid(model.grid_size), num_tests=100))

#         print(f"Epoch {epoch}: Loss: {avg_loss}, Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracies[-1]}")

#     # Plotting
#     plt.figure(figsize=(15, 5))
#     plt.subplot(1, 3, 1)
#     plt.plot(losses, marker='o', linestyle='-')
#     plt.title('Loss')
#     plt.subplot(1, 3, 2)
#     plt.plot(train_accuracies, marker='o', linestyle='-')
#     plt.title('Training Accuracy')
#     plt.subplot(1, 3, 3)
#     plt.plot(test_accuracies, marker='o', linestyle='-')
#     plt.title('Test Accuracy')
#     plt.tight_layout()
#     plt.show()
    

# if __name__ == "__main__":
#     # Model Initialization
#     grid_size = 4
#     cnn_model = CNNModel(grid_size)
#     dataset = generate_dataset(grid_size, num_pairs=1000)
#     train_and_plot(cnn_model, dataset, epochs=100)

#     # Save the model
#     torch.save(cnn_model.state_dict(), f"cnn_model_{grid_size}x{grid_size}.pt")
import torch
from simulator.game15 import *
from util.enums import *
from util.utils import preprocess_state, generate_dataset
from cnn_model import CNNModel
import matplotlib.pyplot as plt

def run_tests(model, grid_size, num_tests=1000, shuffle_distance=10, step_limit=100):
    successful_tests = 0

    for _ in range(num_tests):
        env = Grid(grid_size)
        env.shuffle_n(shuffle_distance)

        state = env.get_state()
        steps = 0

        while not env.is_solved() and steps < step_limit:
            steps += 1

            state_tensor = preprocess_state(state, grid_size).to(model.device)

            action_logits = model(state_tensor)
            action = torch.argmax(action_logits).item() + 1

            env.process_action(action)
            state = env.get_state()

        if env.is_solved():
            successful_tests += 1

    return successful_tests / num_tests

def train_and_log(model, dataset, epochs, learning_rate=0.001, logging_interval=1000):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    losses = []
    train_accuracies = []
    test_accuracies_3x3 = []
    test_accuracies_4x4 = []

    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        total_loss = 0
        correct_train = 0

        for state, action in dataset:
            state_tensor = preprocess_state(state, model.grid_size).to(model.device)
            target_action = torch.tensor([action.value - 1], dtype=torch.long).to(model.device)

            action_logits = model(state_tensor)
            loss = criterion(action_logits, target_action)

            if torch.argmax(action_logits) == target_action:
                correct_train += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataset)
        train_accuracy = correct_train / len(dataset)

        losses.append(avg_loss)
        train_accuracies.append(train_accuracy)

        # Every 1000 epochs, log success rates for both grids
        if (epoch + 1) % logging_interval == 0:
            success_rate_3x3 = run_tests(model, grid_size=3, num_tests=1000)
            success_rate_4x4 = run_tests(model, grid_size=4, num_tests=1000)

            test_accuracies_3x3.append((epoch, success_rate_3x3))
            test_accuracies_4x4.append((epoch, success_rate_4x4))

            print(f"Epoch {epoch}: Loss: {avg_loss}, Train Accuracy: {train_accuracy}")
            print(f"3x3 Success Rate: {success_rate_3x3}, 4x4 Success Rate: {success_rate_4x4}")

    return losses, train_accuracies, test_accuracies_3x3, test_accuracies_4x4

# Training and logging
if __name__ == "__main__":
    grid_size = 4
    cnn_model = CNNModel(grid_size)
    numofepochs = 10000
    dataset = generate_dataset(grid_size, num_pairs=1000*numofepochs)
    losses, train_accuracies, test_accuracies_3x3, test_accuracies_4x4 = train_and_log(cnn_model, dataset, epochs=10000)

    # Save the model
    torch.save(cnn_model.state_dict(), f"cnn_model_{grid_size}x{grid_size}.pt")

    # Display logged results
    print("3x3 Success Rates:", test_accuracies_3x3)
    print("4x4 Success Rates:", test_accuracies_4x4)
