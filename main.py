import pandas as pd
from BayesNet import *
from treelib import Tree
import random

T, F = True, False

df = pd.read_csv("Iris.csv")

print("                     COMPUTATION OF PROBABILITIES")
print("-----------------------------------------------------------------------")

# specie
df_species_x = df.query("Species=='Iris-setosa'")
p_species_x = df_species_x.shape[0] / df.shape[0]
print("P(Iris-setosa): ", p_species_x)

df_species_y = df.query("Species=='Iris-versicolor'")
p_species_y = df_species_y.shape[0] / df.shape[0]
print("P(Iris-versicolor): ", p_species_y)

# sepal length
df_max_sepal_len = df.query("SepalLengthCm>=5.8")
p_max_sepal_len = df_max_sepal_len.shape[0] / df.shape[0]
print("P(SepalL>=5.8): ", p_max_sepal_len)

df_min_sepal_len = df.query("SepalLengthCm<5.8")
p_min_sepal_len = df_min_sepal_len.shape[0] / df.shape[0]
print("P(SepalL<5.8): ", p_min_sepal_len)

df_min_sepal_wid = df.query("SepalWidthCm<3")
p_min_sepal_wid = df_min_sepal_wid.shape[0] / df.shape[0]
print("P(SepalW<3): ", p_min_sepal_wid)

df_max_sepal_wid = df.query("SepalWidthCm>=3")
p_max_sepal_wid = df_max_sepal_wid.shape[0] / df.shape[0]
print("P(SepalW>=3): ", p_max_sepal_wid)

# P(sepalLength>=5.8 | Iris Setosa)
df_max_sepal_x = df.query("SepalLengthCm>=5.8 and Species=='Iris-setosa'")
p_max_sepal_x = (df_max_sepal_x.shape[0] / df.shape[0]) / p_species_x
print("P(sepalLength>=5.8 | Iris Setosa): ", p_max_sepal_x)

# P(sepalLength<5.8 | Iris Setosa)
df_min_sepal_x = df.query("SepalLengthCm<5.8 and Species=='Iris-setosa'")
p_min_sepal_x = (df_min_sepal_x.shape[0] / df.shape[0]) / p_species_x
print("P(sepalLength<5.8 | Iris Setosa): ", p_min_sepal_x)

# P(sepalLength>=5.8 | Iris versicolor)
df_max_sepal_y = df.query("SepalLengthCm>=5.8 and Species=='Iris-versicolor'")
p_max_sepal_y = (df_max_sepal_y.shape[0] / df.shape[0]) / p_species_y
print("P(sepalLength>=5.8 | Iris versicolor): ", p_max_sepal_y)

# P(SepalWidthCm>=3 | SepalLengthCm>=5.8)
df_max_sepal_length_width = df.query("SepalWidthCm>=3 and SepalLengthCm>=5.8")
p_max_sepal_length_width = (df_max_sepal_length_width.shape[0] / df.shape[0]) / p_max_sepal_len
print("P(SepalWidthCm>=3 | SepalLengthCm>=5.8): ", p_max_sepal_length_width)

# P(SepalWidthCm<3 | SepalLengthCm>=5.8)
df_min_sepal_width_length = df.query("SepalWidthCm<3 and SepalLengthCm>=5.8")
p_min_sepal_width_length = (df_min_sepal_width_length.shape[0] / df.shape[0]) / p_min_sepal_wid
print("P(SepalWidthCm<3 | SepalLengthCm>=5.8): ", p_min_sepal_width_length)

# P(SepalWidthCm<3 | SepalLengthCm<5.8)
df_false_sepal_width_length = df.query("SepalWidthCm<3 and SepalLengthCm<5.8")
p_false_sepal_width_length = (df_false_sepal_width_length.shape[0] / df.shape[0]) / p_min_sepal_wid
print("P(SepalWidthCm<3 | SepalLengthCm<5.8): ", p_false_sepal_width_length)

# P(SepalWidthCm>=3 | SepalLengthCm<5.8)
df_min_sepal_length_width = df.query("SepalWidthCm>=3 and SepalLengthCm<5.8")
p_min_sepal_length_width = (df_min_sepal_length_width.shape[0] / df.shape[0]) / p_min_sepal_len
print("P(SepalWidthCm>=3 | SepalLengthCm<5.8): ", p_min_sepal_length_width)

# P(PetalLengthCm>=3.7 | SepalLengthCm>=5.8)
df_max_petal_length = df.query("PetalLengthCm>=3.7 and SepalLengthCm>=5.8")
p_max_petal_length = (df_max_petal_length.shape[0] / df.shape[0]) / p_max_sepal_len
print("P(PetalLengthCm>=3.7 | SepalLengthCm>=5.8): ", p_max_petal_length)

# P(PetalLengthCm>=3.7 | SepalLengthCm<5.8)
df_min_petal_length = df.query("PetalLengthCm>=3.7 and SepalLengthCm<5.8")
p_min_petal_length = (df_min_petal_length.shape[0] / df.shape[0]) / p_min_sepal_len
print("P(PetalLengthCm>=3.7 | SepalLengthCm<5.8): ", p_min_petal_length)

# *******************************
# *  Bayes net implementation   *
# *******************************

irisNet = BayesNet([
    ('Setosa', '', p_species_x),
    ('Sepal_L_Top', 'Setosa', {T: p_max_sepal_x, F: p_max_sepal_y}),
    ('Sepal_W_Top', 'Sepal_L_Top', {T: p_max_sepal_length_width, F: p_min_sepal_length_width}),
    ('Petal_L_Top', 'Sepal_L_Top', {T: p_max_petal_length, F: p_min_petal_length})
])

# Graphing the network to the console
print("\n Bayesian Network")
print("----------------------")
tree = Tree()

tree.create_node("Setosa", "Setosa")
tree.create_node("Sepal_L_Top", "Sepal_L_Top", parent="Setosa")
tree.create_node("Sepal_W_Top", "Sepal_W_Top", parent="Sepal_L_Top")
tree.create_node("Petal_L_Top", "Petal_L_Top", parent="Sepal_L_Top")
tree.show()

# enumeration examples:
print("\n                 ENUMERATION EXAMPLES")
print("------------------------------------------------------------")

print("Given Setosa=T --Sepal_W_Top ", enumeration_ask(
    'Sepal_W_Top', dict(Setosa=T),
    irisNet).show_approx(), "\n")

print("Given Sepal_L_Top=T --Petal_L_Top", enumeration_ask(
    'Petal_L_Top', dict(Sepal_L_Top=T),
    irisNet).show_approx())

# variable elimination examples:
print("\n             VARIABLE ELIMINATION EXAMPLES")
print("------------------------------------------------------------")

print("P(Setosa|Sepal_LTop=T, Sepal_W_Top=T): ", elimination_ask(
    'Setosa', dict(Sepal_L_Top=T, Sepal_W_Top=T),
    irisNet).show_approx(), "\n")

print("P(Setosa|Sepal_W_Top=T, Petal_L_Top=T): ", elimination_ask(
    'Setosa', dict(Sepal_W_Top=T, Petal_L_Top=T),
    irisNet).show_approx())

# rejection sampling examples:
print("\n              REJECTION SAMPLING EXAMPLES")
print("------------------------------------------------------------")

print("P(Setosa|Sepal_LTop=T, Sepal_W_Top=T): ", rejection_sampling(
    'Setosa', dict(Sepal_L_Top=T, Sepal_W_Top=T),
    irisNet, 10000).show_approx(), "\n")

print("P(Setosa|Sepal_W_Top=T, Petal_L_Top=T): ", rejection_sampling(
    'Setosa', dict(Sepal_W_Top=T, Petal_L_Top=T),
    irisNet, 10000).show_approx())

# likelihood weighting sampling examples:
print("\n            LIKELIHOOD WEIGHTING EXAMPLES")
print("------------------------------------------------------------")

print("P(Setosa|Sepal_LTop=F, Sepal_W_Top=F): ", likelihood_weighting(
    'Setosa', dict(Sepal_L_Top=F, Sepal_W_Top=F),
    irisNet, 10000).show_approx(), "\n")

print("P(Setosa|Sepal_W_Top=T, Sepal_L_Top=T): ", likelihood_weighting(
    'Setosa', dict(Sepal_W_Top=T, Sepal_L_Top=T),
    irisNet, 10000).show_approx())

# prior sampling examples:
print("\n               PRIOR SAMPLING EXAMPLES")
print("------------------------------------------------------------")

random.seed(128)
all_obs = [prior_sample(irisNet) for x in range(1000)]
Sepal_W_Top_true = [observation for observation in all_obs if observation['Sepal_W_Top'] == True]
Petal_L_Top_true = [observation for observation in all_obs if observation['Petal_L_Top'] == True]
print("\nP(Sepal_W_Top=T):", len(Sepal_W_Top_true) / 1000)
print("\nP(Petal_L_Top=T):", len(Petal_L_Top_true) / 1000)
