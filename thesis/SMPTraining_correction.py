# 1. train the model
for epoch in range(num_epochs):
    model.train() # set model to training mode
    for inputs, labels in train_loader:
        # move input and label tensors to the device
        inputs = inputs.to(device)
        labels = labels.to(device)
        # zero out the optimizer
        optimizer.zero_grad()
        # forward pass
        outputs = model(inputs)
        # loss is computed differently in first epoch
        if epoch == 0:
            loss = criterion(outputs, labels)
        else:
            # 2.2 label correction
            pseudo_labels = []
            for inputs, labels in train_loader:
                # reset intermediate_output
                reset_intermediate_output()
                model(inputs)
                # extract features of the samples with the hook 
                # (automatically saved as intermediate_output)
                batch_pseudo_labels = correct_labels(prototypes, 
                    intermediate_output, labels)
                pseudo_labels.extend(batch_pseudo_labels)
            # calculate the loss on pseudo and original labels
            loss_original = (1-alpha) * criterion(outputs, labels)
            loss_pseudo = alpha * criterion(outputs, pseudo_labels)
            loss = loss_original + loss_pseudo
