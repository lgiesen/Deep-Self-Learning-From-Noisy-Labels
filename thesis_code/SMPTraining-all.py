# 1. train the model
for epoch in range(num_epochs):
    model.train() # Set model to training mode
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
                # (automatically in the intermediate_output variable)
                batch_pseudo_labels = correct_labels(prototypes, 
                    intermediate_output, labels)
                pseudo_labels.extend(batch_pseudo_labels)
            # calculate the loss on pseudo and original labels
            loss_original = (1-alpha) * criterion(outputs, labels)
            loss_pseudo = alpha * criterion(outputs, pseudo_labels)
            loss = loss_original + loss_pseudo
    # 2. label correction phase (except in last iteration)
    with torch.no_grad():
        if epoch != num_epochs-1:
            # 2.1 prototype selection
            # initialize prototypes
            prototypes = []
            for class_id in range(num_classes):
                # Reset the features
                reset_intermediate_output()
                # sample images for the current class
                sample_loader = sample_images(class_id)
                # feed the randomly sampled images through the model to extract
                # the features of the samples with the hook
                # (automatically in the intermediate_output variable)
                samples_features = []
                for inputs in sample_loader:
                    inputs = inputs.to(device)
                    model(inputs)
                    samples_features.append(intermediate_output)
                    reset_intermediate_output()
                # convert to tensor for following operations
                samples_features = torch.cat(samples_features)
                # calculate similarity matrix of sample features
                similarity_matrix = cos_similarity(samples_features)
                # calculate the threshold
                threshold = calc_similarity_threshold(similarity_matrix)
                # calculate density of images
                densities = calc_rho_densities(similarity_matrix, threshold)
                # calculate similarity measurement
                similarity_measurement = calc_eta_similarity_measurement(
                    similarity_matrix, densities)
                # select prototypes for each class
                class_prototypes = select_prototypes(
                    similarity_measurement, samples_features)
                prototypes.append(class_prototypes)
            prototypes = torch.cat(prototypes)
    scheduler.step()