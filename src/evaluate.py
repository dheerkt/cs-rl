def load_trained_agents(checkpoint_path, device='cpu'):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Infer obs_dim from checkpoint's first Linear layer weight shape
    in_features = checkpoint['actor0_state_dict']['shared.0.weight'].shape[1]
    
    actors = [
        ActorNetwork(
            obs_dim=in_features,  # Use checkpoint's dimension
            action_dim=HyperParams.action_dim,
            hidden_size=HyperParams.hidden_size,
            num_layers=HyperParams.num_layers
        ).to(device),
        ActorNetwork(
            obs_dim=in_features,
            action_dim=HyperParams.action_dim,
            hidden_size=HyperParams.hidden_size,
            num_layers=HyperParams.num_layers
        ).to(device)
    ]
    
    actors[0].load_state_dict(checkpoint['actor0_state_dict'])
    actors[1].load_state_dict(checkpoint['actor1_state_dict'])
    actors[0].eval()
    actors[1].eval()
    
    print(f"Loaded checkpoint from {checkpoint_path} with obs_dim={in_features}")
    return actors
