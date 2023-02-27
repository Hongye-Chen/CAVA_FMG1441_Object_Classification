import pandas as pd
import os
import random
import yaml

images_trial = 20
trials_block = 24
total_blocks = 10
image_repeat_high = 8
image_repeat_low = 4
# altha = 0.2 #Percentage of images that gonna be repeated more
unique_images = 975
target_images = 25
trial_matrix = []       #Initiate the whole matrix
total_trials = trials_block * total_blocks
files_path = os.path.join("C:\\", "Users", "15202", "OneDrive", "C_", "University of Amsterdam", "Intern")

high_repeat_range = 195  # How many images need to be presented more (9 times)      #int(altha * unique_images)
total_targets = 120  # int(target_images * (high_per * image_repeat_high + (1-high_per) * image_repeat_low))

images_list = list(range(unique_images))     # The list of unique images
random.shuffle(images_list)     # Shuffle the list, the first 199 images gonna be presented 9 times

images_rep_list = [image_repeat_low] * unique_images   # A list to track how many times remaining for each image to be presented, 4 times for most of the images
for i in range(high_repeat_range):
    images_rep_list[images_list[i]] = image_repeat_high   # The first 195 images gonna be presented 8 times
images_rep_list = images_rep_list + [1] * total_targets   # Each target image only present 1 time

for i in range(total_trials):      # Start Randomly assigning images for each trial
    current_trial = []             #Initiate the image list for current trial
    images_track = [1] * (unique_images+total_targets)      #Track with which images has been presented in the current trial
    for j in range(images_trial):   # In the current trial
        iteration_time = 0          # Track with how many times the program has iterated
        remaining_images = [ind for ind, val in enumerate(images_rep_list) if val > 0]      #Which images have not been run out
        cur_image = random.choice(remaining_images)         #Randomly choose a image from the remaining
        while images_track[cur_image] < 1 or images_rep_list[cur_image] <=0: #If the image has been choosen in this trial, choose another image
            # print(cur_image)
            cur_image = random.choice(remaining_images)
            iteration_time += 1
            if iteration_time > 1500:   #If iterated too many times, quit the iteration
                break
        if iteration_time > 1500:
            break
        current_trial.append(cur_image)     #   if the image meet the requirement, add it into current trial
        images_track[cur_image] -= 1        #   Record that this image has been used in the current trial
        images_rep_list[cur_image] -= 1     #   Update the remained time
        if cur_image >= unique_images:
            images_track[unique_images:] = [-1] * total_targets     #If a target image was chose, other target images can not be chose in the current trial
    if iteration_time > 1500:
        break
    trial_matrix.append(current_trial)  #Add the complete trial into the matrix

for i in remaining_images:      #For the remaining images, switch them with other images in the previous trials to complete the matrix
    while images_rep_list[i] > 0:
        trial_to_switch = random.randint(0,total_trials-2)
        image_to_switch = random.randint(0,19)
        while i in trial_matrix[trial_to_switch] and not(trial_matrix[trial_to_switch][image_to_switch] in current_trial):
            trial_to_switch = random.randint(0, total_trials-2)
            image_to_switch = random.randint(0, 19)
        image_stored = trial_matrix[trial_to_switch][image_to_switch]
        trial_matrix[trial_to_switch][image_to_switch] = i
        current_trial.append(image_stored)
        images_rep_list[i] -= 1

trial_matrix.append(current_trial)

no_repeat = True
for i in trial_matrix:      #Double check if there are repeated images in each trial
    no_repeat = no_repeat and (len(i) == len(set(i)))
    if not no_repeat:
        break

print("No repeated images in each trial: ", no_repeat)


images_population = [0] * (unique_images + total_targets)   #Check the images population, it should be 1,4,8
for i in range(unique_images + total_targets):
    for j in trial_matrix:
        if i in j:
            images_population[i] += 1

print("Images population:", set(images_population))

correct_answer = []
for i in trial_matrix:
    for j in range(images_trial):        #number of images per trial
        if i[j] > unique_images-1:          #In our case, if the number is bigger than 975, it's a target, assign 'k'
            correct_answer.append('k')
            break
        elif j==(images_trial-1):             #If the last image is still not a target, assign 'l'
            correct_answer.append('l')
            break

correct_answer_data = {'correct_answers' : correct_answer}      #save the dataframe to a csv file
correct_answer_dataframe = pd.DataFrame(correct_answer_data)
correct_answer_dataframe.to_csv(os.path.join(files_path, "Correct_answers_975.csv"),index=False, header = None)

yaml_path = os.path.join(files_path, "eeg_oads_stimulus_filenames.yml")
with open(yaml_path, 'rb') as f:
    subjects = yaml.load(f, Loader=yaml.UnsafeLoader)
current_subject = subjects["sub_0"]
for i in range(total_trials):
    for j in range(images_trial):
        if trial_matrix[i][j] > unique_images:
            trial_matrix[i][j] = random.choice(current_subject[unique_images: unique_images+target_images])
        else:
            trial_matrix[i][j] = current_subject[trial_matrix[i][j]]


randomized_matrix_dataframe = pd.DataFrame(trial_matrix)
randomized_matrix_dataframe.to_csv(os.path.join(files_path, "randomized_matrix_975.csv"), index=False, header = None)






