def prepare_features(input_dict, current_year=2024) -> dict:
    """
    Generate new features based on the input dictionary and return an updated dictionary.
    
    Args:
        input_dict (dict): A dictionary containing the original features.
        current_year (int): The current year for calculating the house age.
    
    Returns:
        dict: A dictionary updated with the newly generated features.
    """
    # Copy the original dictionary to avoid mutating the input
    updated_dict = input_dict.copy()
    
    # Calculate new features
    updated_dict['house_age'] = current_year - updated_dict['year_built']
    updated_dict['bed_bath_ratio'] = updated_dict['num_bedrooms'] / updated_dict['num_bathrooms']
    updated_dict['lot_size_per_sqft'] = updated_dict['lot_size'] / updated_dict['square_footage']
    updated_dict['garage_space_per_bedroom'] = updated_dict['garage_size'] / updated_dict['num_bedrooms']
    updated_dict['recently_renovated'] = int(updated_dict['house_age'] <= 10)
    updated_dict['modernness_index'] = (
        (10 - updated_dict['house_age'] / 10) 
        + updated_dict['garage_size'] 
        + updated_dict['neighborhood_quality']
    )
    updated_dict['outdoor_space'] = updated_dict['lot_size'] - (updated_dict['square_footage'] / 43560)  # 43560 sq. ft. in an acre
    updated_dict['expansion_potential'] = updated_dict['lot_size'] * updated_dict['neighborhood_quality']
    
    return updated_dict
