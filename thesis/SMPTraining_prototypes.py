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
            # Feed the randomly sampled images through the model
            # to automatically extract features into the 
            # intermediate_output variable using the hook
            samples_features = []
            for inputs in sample_loader:
                inputs = inputs.to(device)
                model(inputs)
                samples_features.append(intermediate_output)
                reset_intermediate_output()
            # convert to tensor for following operations
            samples_features = torch.cat(samples_features)
            # calculate similarity matrix of sample features
            similarity_matrix = cos_similarity(
                samples_features, randomly_sampled_img_count)
            # calculate the threshold
            threshold = calc_similarity_threshold(
                similarity_matrix, threshold_percentile)
            # calculate density of images
            densities = calc_rho_densities(
                similarity_matrix, threshold)
            # calculate similarity measurement
            eta = calc_eta_similarity_measurement(
                similarity_matrix, densities)
            # select prototypes for each class
            class_prototypes = select_prototypes(
                eta, samples_features, num_prototypes)
            prototypes.append(class_prototypes)
        prototypes = torch.cat(prototypes)
scheduler.step()