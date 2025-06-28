import os

print("=== Checking folder contents ===")

# Check good folder
good_folder = "data/good"
if os.path.exists(good_folder):
    good_files = os.listdir(good_folder)
    print(f"\nGood folder ({len(good_files)} files):")
    for f in good_files:
        print(f"  - {f}")

# Check clearence_problems folder
prob_folder = "data/clearance_problems"
if os.path.exists(prob_folder):
    prob_files = os.listdir(prob_folder)
    print(f"\nClearence_problems folder ({len(prob_files)} files):")
    for f in prob_files:
        print(f"  - {f}")
else:
    print(f"\nFolder {prob_folder} doesn't exist!")

print("\n=== Ready to test detector ===")