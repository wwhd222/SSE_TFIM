import argparse
from qmc_tfim.main import groundstate, mixedstate

def get_user_input(prompt, default=None, type_func=str):
    if default is not None:
        prompt = f"{prompt} [{default}]: "
    else:
        prompt = f"{prompt}: "
    
    while True:
        user_input = input(prompt)
        if user_input == "" and default is not None:
            return default
        try:
            return type_func(user_input)
        except ValueError:
            print(f"Invalid input. Please enter a valid {type_func.__name__}.")

def main():
    print("Welcome to the QMC TFIM Simulation Runner")
    print("=========================================")

    # Choose simulation type
    sim_type = get_user_input("Choose simulation type (groundstate/mixedstate)", default="groundstate")

    # Get common parameters
    dims = [int(x) for x in get_user_input("Enter lattice dimensions (space-separated integers)", default="10 10").split()]
    periodic = get_user_input("Use periodic boundary conditions? (y/n)", default="y").lower() == "y"
    field = get_user_input("Enter transverse field strength", default=1.0, type_func=float)
    interaction = get_user_input("Enter interaction strength", default=1.0, type_func=float)
    M = get_user_input("Enter half-size of operator list", default=1000, type_func=int)
    measurements = get_user_input("Enter number of measurements", default=100000, type_func=int)
    skip = get_user_input("Enter number of steps to skip between measurements", default=0, type_func=int)

    # Create argument dictionary
    args = {
        "dims": dims,
        "periodic": periodic,
        "field": field,
        "interaction": interaction,
        "M": M,
        "measurements": measurements,
        "skip": skip
    }

    # Add beta for mixedstate simulation
    if sim_type == "mixedstate":
        beta = get_user_input("Enter inverse temperature (beta)", default=10.0, type_func=float)
        args["beta"] = beta

    # Create Namespace object
    args = argparse.Namespace(**args)

    # Run simulation
    print("\nStarting simulation...")
    if sim_type == "groundstate":
        groundstate(args)
    else:
        mixedstate(args)
    
    print("Simulation completed!")

if __name__ == "__main__":
    main()
