import pandas as pd
import os
import random

images_trial = 20
trials_block = 10
blocks_repetition = 6
repetition = 4
image_repeat_high = 8
image_repeat_low = 4
high_per = 0.2
unique_images = 995
target_images = 5
trial_matrix = []       #Initiate the matrix
total_trials = trials_block * blocks_repetition * repetition
files_path = os.path.join("C:\\", "Users", "15202", "OneDrive", "C_", "University of Amsterdam", "Intern")

high_repeat_range = int(high_per * unique_images) # How many images need to be presented more (8 times)
total_targets = int(target_images * (high_per * image_repeat_high + (1-high_per) * image_repeat_low))

images_list = list(range(unique_images))
random.shuffle(images_list)

images_rep_list = [image_repeat_low] * unique_images   #Record how many times remaining for each image to be presented
for i in range(high_repeat_range):
    images_rep_list[images_list[i]] = image_repeat_high
images_rep_list = images_rep_list + [1] * total_targets

for i in range(total_trials):      #Randomly choose images and assign them to the matrix
    current_trial = []             #Initiate the image list for current trial
    images_track = [1] * (unique_images+total_targets)      #Track with which images has been presented in current trial
    for j in range(images_trial):
        iteration_time = 0
        remaining_images = [ind for ind, val in enumerate(images_rep_list) if val > 0]      #Which images still have not been run out
        cur_image = random.choice(remaining_images)         #Randomly choose a image from the remaining
        while images_track[cur_image] < 1 or images_rep_list[cur_image] <=0: #If the image has been choosen in this trial, choose again
            # print(cur_image)
            cur_image = random.choice(remaining_images)
            iteration_time += 1
            if iteration_time > 1500:   #If iterating too many times, meaning no solution, quit the iteration
                break
        if iteration_time > 1500:
            break
        current_trial.append(cur_image)     #if the image meet the requirement, add it into the trial
        images_track[cur_image] = images_track[cur_image] - 1
        images_rep_list[cur_image] = images_rep_list[cur_image] - 1
        if cur_image >= unique_images:
            images_track[unique_images:] = [-1] * total_targets
    if iteration_time > 1500:
        break
    trial_matrix.append(current_trial)  #Add the complete trial into the matrix

for i in remaining_images:      #For the remaining images, switch them with other images in the previous trials to complete the matrix
    while images_rep_list[i] > 0:
        trial_to_switch = random.randint(0,238)
        image_to_switch = random.randint(0,19)
        while i in trial_matrix[trial_to_switch] and not(trial_matrix[trial_to_switch][image_to_switch] in current_trial):
            trial_to_switch = random.randint(0, 238)
            image_to_switch = random.randint(0, 19)
        image_stored = trial_matrix[trial_to_switch][image_to_switch]
        trial_matrix[trial_to_switch][image_to_switch] = i
        current_trial.append(image_stored)
        images_rep_list[i] -= 1

no_repeat = True
for i in trial_matrix:      #Double check if there are repeated images in each trial
    no_repeat = no_repeat and (len(i) == len(set(i)))
    if not no_repeat:
        break
print(no_repeat)

randomized_matrix_dataframe = pd.DataFrame(trial_matrix).T
randomized_matrix_dataframe.to_csv(os.path.join(files_path, "randomized_matrix_995.csv"), index=False)

correct_answer = []
for i in trial_matrix:
    for j in range(images_trial):        #number of images per trial
        if i[j] > unique_images-1:          #In our case, if the number is bigger than 994, it's a target, assign 'y'
            correct_answer.append('y')
            break
        elif j==(images_trial-1):             #If the last image (all images) is still not a target, assign 'n'
            correct_answer.append('n')
            break

correct_answer_data = {'correct_answers' : correct_answer}      #save the dataframe to a csv file
correct_answer_dataframe = pd.DataFrame(correct_answer_data)
correct_answer_dataframe.to_csv(os.path.join(files_path, "Correct_answers_995.csv"),index=False, header = None)










