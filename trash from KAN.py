trash from KAN
# Perform automatic symbolic regression
#lib = ['x', 'x^2', 'x^3', 'x^4', 'exp', 'log', 'sqrt', 'tanh', 'sin', 'abs']
#model.auto_symbolic(lib=lib)

# Get the symbolic formulas
#formula1, formula2 = model.symbolic_formula()[0]

# Define a function to compute the accuracy of the formulas
def acc(formula1, formula2, loader):
    correct = 0
    total = 0
    for images, labels in loader:
        batch = images.shape[0]
        for i in range(batch):
            logit1 = np.array(formula1.subs({'x_1': images[i, 0], 'x_2': images[i, 1]})).astype(np.float64)
            logit2 = np.array(formula2.subs({'x_1': images[i, 0], 'x_2': images[i, 1]})).astype(np.float64)
            correct += (logit2 > logit1) == labels[i]
            total += 1
    return correct / total

# Compute the accuracy of the formulas on the train and test sets
#print('Train accuracy of the formula:', acc(formula1, formula2, train_loader))
#print('Test accuracy of the formula:', acc(formula1, formula2, test_loader))

def test_model(model, criterion, dataloader):
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():  # No need to track gradients during testing
        for inputs_test, labels_test in dataloader:
            print(f"Input shape: {inputs_test.shape}, Label shape: {labels_test.shape}")
            print(inputs_test)
            print(labels_test)
            outputs = model(inputs_test)
            _, predicted_labels = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels_test)
            
            # Update running loss
            running_loss += loss.item()
            
            # Count correct predictions
            correct_predictions += (predicted_labels == labels_test).sum().item()
            total_samples += labels_test.size(0)
    
    # Calculate average loss and accuracy
    avg_loss = running_loss / len(dataloader)
    accuracy = correct_predictions / total_samples
    
    print(f'Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}')

# Training loop
def train(model, optimizer, criterion, train_loader, steps):
    for epoch in range(num_epochs): # loop over the dataset multiple times
            running_loss = 0.0
            for i, (inputs, targets) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()                
                # Optionally, print statistics
                running_loss += loss.item()
                print(f'Epoch: {epoch+1}, Loss: {loss.item()}, Percent Complete: {i/len(train_loader)}')
            avg_loss = running_loss / len(train_loader)
            print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')




