import matplotlib.pyplot as plt
import os
import numpy as np
import ast
import seaborn as sns
		
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.figure(figsize=(10, 5))

newRewards = []
if os.path.exists("Records"):
	f=open("./Records/q_val.txt","r")
	string=f.readline()
	newRewards = []
	while string!="":
		if string != "":
			string = ast.literal_eval(string)
			newRewards.extend(string)
		string=f.readline()

	newRewards =  np.array(newRewards).reshape(len(newRewards), 1)
	f.close()

	sns.set()
	sns.set_palette("coolwarm")
	plt.semilogy(newRewards, linewidth=1)	   
	plt.xlabel("Frame", weight='bold')
	plt.ylabel("Value Estimate", weight='bold')

	plt.show() 
