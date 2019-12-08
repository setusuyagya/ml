import pandas as pd
import numpy as np
data = pd.DataFrame()
data['Gender'] = ['male','male','male','male','female','female','female','female']
data['Height'] = [6,5.92,5.58,5.92,5,5.5,5.42,5.75]
data['Weight'] = [180,190,170,165,100,150,130,150]
data['Foot_Size'] = [12,11,12,10,6,8,7,9]
print('\nDataset')
print(data)
person = pd.DataFrame()
person['Height'] = [5]
person['Weight'] = [130]
person['Foot_Size'] = [6]
print('\nTest Instance')
print(person)
n_male = data['Gender'][data['Gender'] == 'male'].count()
n_female = data['Gender'][data['Gender'] == 'female'].count()
total_ppl = data['Gender'].count()
P_male = n_male/total_ppl
P_female = n_female/total_ppl
data_means = data.groupby('Gender').mean()
print('\nDataset Mean')
print(data_means)
data_variance = data.groupby('Gender').var()
print('\nDataset Variance')
print(data_variance)
male_height_mean = data_means['Height'][data_variance.index == 'male'].values[0]
male_weight_mean = data_means['Weight'][data_variance.index == 'male'].values[0]
male_footsize_mean = data_means['Foot_Size'][data_variance.index == 'male'].values[0]
male_height_variance = data_variance['Height'][data_variance.index == 'male'].values[0]
male_weight_variance = data_variance['Weight'][data_variance.index == 'male'].values[0]
male_footsize_variance = data_variance['Foot_Size'][data_variance.index == 'male'].values[0]
female_height_mean = data_means['Height'][data_variance.index == 'female'].values[0]
female_weight_mean = data_means['Weight'][data_variance.index == 'female'].values[0]
female_footsize_mean = data_means['Foot_Size'][data_variance.index == 'female'].values[0]
female_height_variance = data_variance['Height'][data_variance.index == 'female'].values[0]
female_weight_variance = data_variance['Weight'][data_variance.index == 'female'].values[0]
female_footsize_variance = data_variance['Foot_Size'][data_variance.index == 'female'].values[0]
def p_x_given_y(x, mean_y, variance_y):
    p = 1/(np.sqrt(2*np.pi*variance_y)) * np.exp((-(x-mean_y)**2)/(2*variance_y))
    return p

print("\nProbability Male")
prob_male=P_male * \
p_x_given_y(person['Height'][0], male_height_mean, male_height_variance) * \
p_x_given_y(person['Weight'][0], male_weight_mean, male_weight_variance) * \
p_x_given_y(person['Foot_Size'][0], male_footsize_mean, male_footsize_variance)
print(prob_male)
print("\nProbability Female")
prob_female=P_female * \
p_x_given_y(person['Height'][0], female_height_mean, female_height_variance) * \
p_x_given_y(person['Weight'][0], female_weight_mean, female_weight_variance) * \
p_x_given_y(person['Foot_Size'][0], female_footsize_mean,female_footsize_variance)
print(prob_female)
if(prob_male>prob_female):
    print("\nTarget Label: Male")
else:
    print("\nTarget Label: Female")