#!/usr/bin/env python3
#-*- coding: utf-8 -*-
import os
current_path = f"{os.path.dirname(os.path.abspath(__file__))}/"
'''----------------------------------------------------------------------------------------
Copyright© 2019 Jules Devaux All Rights Reserved
This script may not be copied, altered, distributed or processed by any rights,
unless granted by the owner (i.e. Jules Devaux)
-------------------------------------------------------------------------------------------

DatLabV2.py

A whole new way to extract DatLab results.

Requires:
	- Data file:
		- a folder containing all csv files or:
		- a csv file exported from DatLab
			- Time: in seconds
			- [O2]
			- JO2 normalised per mass: if not, precise in the parameter section
					it will retrieve mass from repository and correct JO2,.
			- Block T° in Celsius
			- Events
			- Fluo: optional
	- Modify the extract_

Optional:
	- Repository that contains (for each experiment):
		- name of DLD folder
		- sample number
		- sample mass for each chamber "A_mass, B_mass"
		- experimental condition if any
		=> all info in repo will be appended to each chamber as columns
'''

#=================================== PARAMETERS ===========================================

# Data:
DATA_PATH = f"{current_path}Data/Raw/"
CSV_PATH = f"{current_path}CSV/"

# Experimental:
TEMPERATURES = [18,30]
PROTOCOLS={'ADP':['MgG', 'MgCl2_1','MgCl2_2', 'Bleb', 'Ouab', 'Heart', 'PMG', 'ADP1', 'ADP2', 'S', 'Oli', 'EGTA'],
		'ATP':['MgG', 'MgCl2_1','MgCl2_2', 'Heart', 'PMG', 'ATP1', 'ATP2', 'S', 'Oli', 'Bleb', 'Ouab', 'EGTA']}

Mgfree_calib = {'MgG':1,
				'EGTA':0,
				'MgCl2_1':1.625,
				'MgCl2_2':2.25}

ATP_calib = {
			'PMG':0,

			'ATP1':0.625,
			'ATP2':1.25,

			'ADP1':0.625,
			'ADP2':1.25}

chamber_V = 2 #ml

# Graph:
graphing = False
to_plot = ['B_O2', 'B_JO2', 'B_ATP'] # 
x_range = None # Can choose 'start' or as format: (start) / (start,end)

# Constants:
F = 96485.33212 #C mol−1
R = 8.314472 #J.K-1
Z = 1 #Valence of TMR(M/E) / Safr
#=========================================================================================





#////////////////////////////////////// SCRIPT \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn import linear_model
import itertools



class Graph():

	def __init__(self):
		pass

	def set_line_color(self, y_label, color=None):
		if color != None:
			return color
		elif "JO2" in y_label:
			color = '#aa0032'
		elif "O2" in y_label:
			color = '#309BCD'
		elif "DYm" in y_label:
			color = "#8100c7"
		elif "ATP" in y_label:
			color = "#00db0f"
		elif "ROS" in y_label:
			color = "#00db0f"
		else:
			if color is None:
				color = "#b0b0b0"
		return color


	def set_Theme(self, ax, y_label=None):
		# A few options:
		ax.grid(True)
		ax.set_facecolor('#333744')
		ax.set_autoscale_on
		ax.spines['bottom'].set_color('#808595')
		ax.spines['top'].set_color('#808595')
		ax.spines['left'].set_color('#808595')
		ax.spines['right'].set_color('#808595')
		ax.tick_params(axis='both', direction='in',
						color='#808595', labelcolor='#808595',
						grid_color='#465063', grid_linestyle='--', grid_linewidth=0.5)
		return ax


	def set_label(self, label=None):
			if ('JO2' in label):
				return 'JO2 (pmolO2/(s*mg))'
			elif ('DYm' in label):
				return 'DYm (U)'
			elif 'date' in label:
				return 'Date - Time'
			else: return 'N/A'


	def set_ax(self, ax, x_y_s):
		x=x_y_s[0]#.to_list()
		x_label=self.set_label(str(x.name))

		#--- sort y ---
		for j in range(1,len(x_y_s)):
			#color=self.set_line_color(y_label)[j]
			if type(x_y_s[j]) is dict:
				# Define label forst
				try:
					if x_y_s[j]['label']:
						label=x_y_s[j]['label']
					else:label=x_y_s[j]['y'].name
				except:label=None
				y=x_y_s[j]['y'].to_list()
				try: color=self.set_line_color(y_label='None', color=x_y_s[j]['color'])
				except:pass
				if x_y_s[j]['type']=='line':ax.plot(x,y,color=color,linewidth=1,label=label)
				elif x_y_s[j]['type']=='scatter':ax.scatter(x,y,color=color,label=label)
				elif x_y_s[j]['type']=='bar':ax.bar(x,y,color=color,label=label,width=0.5)
			else:ax.plot(x , x_y_s[j].to_list(), color=color, label=x_y_s[j].name)
		ax=self.set_Theme(ax)
		ax.set_xlabel(x_label, labelpad=10, color='#808595', fontsize='small',fontweight='bold')		
		ax.set_ylabel(label, labelpad=10, color='#808595', fontsize='x-small',fontweight='bold')
		ax.legend(loc='upper left', fontsize='xx-small')
		return ax
	

	def graph(self, x_y_s, x_range=None, title=None):
		'''
		Kwargs are passed to matplotlib plotting functions.
		'''
		if type(x_y_s[0]) is pd.core.series.Series:
			x_y_s=[x_y_s]
		subplot_nb=len(x_y_s)
		fig, axs = plt.subplots(subplot_nb, 1, facecolor='#283036', sharex=True)
		#--- one plot (need to otherwise returns error) --
		if subplot_nb==1:
			x_y_s=x_y_s[0]
			axs=self.set_ax(axs,x_y_s)
			
		#--- >1 subplot --- 
		else:
			for i in range(subplot_nb):
				axs[i]=self.set_ax(axs[i],x_y_s[i])
			plt.subplots_adjust(hspace=0)
		if x_range:
			plt.xlim(x_range)
		if title:
			plt.suptitle(title, fontsize=50, fontweight='heavy', color='#808595', alpha=0.2, x=0.5,y=0.5)
		plt.show()


def del_reox(df, o2col='O2', jo2col= 'JO2', _mode='auto', window=[-5,300]):
	'''
	delete reoxygenation if axsked
	'''
	#-- delete reox
	# Smooth O2
	df[o2col]=df[o2col].rolling(window=4).mean()
	if _mode == 'auto':
		df['O2diff']=df[o2col].diff()
		df = df[df['O2diff']<0]
		df=df.set_index('Time [min]'
							).sort_index(
							).reset_index()
		df=df[(df[jo2col]>window[0])&(df[jo2col]<window[1])]

	elif _mode == 'reox':
		df = df[df[f"Event"]!='reox'].sort_index()

	return df


def sort_events(df, additive=True):
	"""
	Distributes events name and comments (text) to respective chambers.
	Takes the (yet) unprocessed df (i.e. exported csv from DatLab)
	"""
	try:
		df['Event Name'] = df['Event Name'].fillna(method='ffill')
		df['Chamber'] = df['Chamber'].fillna(method='ffill')
		df[f'A:_Event'] = np.where((df['Chamber'] == 'Left') | (df['Chamber'] == 'Both'),df['Event Name'], float('nan'))
		df[f'A:_Event_text'] = np.where((df['Chamber'] == 'Left') | (df['Chamber'] == 'Both'),df['Event Text'], float('nan'))
		df[f'B:_Event'] = np.where((df['Chamber'] == 'Right') | (df['Chamber'] == 'Both'),df['Event Name'], float('nan'))
		df[f'B:_Event_text'] = np.where((df['Chamber'] == 'Right') | (df['Chamber'] == 'Both'),df['Event Text'], float('nan'))
		df[f'A:_Event'] = df[f'A:_Event'].fillna(method='ffill')
		df[f'A:_Event_text'] = df[f'A:_Event_text'].fillna(method='ffill')
		df[f'B:_Event'] = df[f'B:_Event'].fillna(method='ffill')
		df[f'B:_Event_text'] = df[f'B:_Event_text'].fillna(method='ffill')
	except Exception as e: print(f"ERROR: sort_events(): {e}")

	return df


def split_chambers(df, rename_cols=True, _index='time', delete_reox='auto', get_weight_col=False, **kwargs):

	"""
	Seperate each chamber to an individual-like experiment

	Expects raw csv from DatLab export.
	
	Overall process:
	- attributes events to each chamber to ease pandas'groupby()' method
	- rename columns if True (instead of 1A:..., gets 'JO2' or parameter e.g. DYm, ROS, ATP)
	- Deletes reoxygenation envents if True (default)

	Each chamber is returned as a dict containing:
			chamber = {
			'chamber': chamber name,
			'protocol': protocol (defined with PROTOCOLS variable),
			'df': df
			}

	"""

	chambers=[]

	for C in ['A', 'B']:
		time_col=[c for c in df.columns if 'Time [' in c][0]
		cols = [time_col] + [c for c in df.columns if f"{C}:" in c]
		cdf=df.loc[:,cols]

		#- define columns
		o2col=[c for c in cols if 'concentration' in c][0]
		jo2col=[c for c in cols if any(w in c for w in ['per mass', 'O2 slope neg'])][0]
		fluo=[c for c in cols if 'Amp' in c or 'pX' in c][0]
		event_col=[c for c in cols if all([f"{C}:_Event" in c, "text" not in c])][0]
		event_text_col=[c for c in cols if all([f"{C}:_Event" in c, "text" in c])][0]

		# Define the protocol
		protocol=None # Default
		if 'A:' in fluo: protocol = 'ADP'
		elif 'B:' in fluo: protocol = 'ATP'

		#-- delete reox
		if delete_reox != None:
			if 'window' in kwargs:
				window=kwargs['window']
			else:window=[-5,300]
			cdf=del_reox(cdf,
							o2col=o2col,
							jo2col=jo2col,
							_mode=delete_reox,
							window=window)

		#-- rename col
		if rename_cols is True:

			coldict={
				time_col: 'time',
				o2col: "O2",
				jo2col: "JO2",
				fluo: protocol,
				event_col: 'Event',
				event_text_col: 'Event_text'
			}

			cdf=cdf.rename(columns=coldict)
			cdf=cdf.set_index(_index
				).sort_index()
			#cols=['time', "JO2", "O2", protocol, "Event", "Event_text"]


		chamber = {
			'chamber': C,
			'protocol': protocol,
			'df': cdf
			}

		#-- retrieve weight
		# Careful not to overwrite repo
		if get_weight_col is not False:
			weight_cols=[c for c in df.columns if get_weight_col in c]
			weight_col=[c for c in weight_cols if C in c][0]
			_mass=df[weight_col].iloc[0]
			chamber.update({'mass': _mass})


		chambers.append(chamber)

	return chambers[0], chambers[1]


def mgfree_calib(chamber, calibration={'MgG':0}):

	mgcalib={}
	for calib, conc in calibration.items():
		raw=chamber['df'].loc[chamber['df']['Event']==calib, chamber['protocol']].mean()
		mgcalib.update({conc:raw})
	plt.plot(*zip(*sorted(mgcalib.items())))
	plt.show()
	print(mgcalib)
	exit()




def sc_ATP(df, adtp='ATP', ATP_calib=None, _mass=1, Mgcalib=True, MgGconc=1.1):
	'''
	Calibrate the ATP signal
	
	Requires df with at least 'Event' and 'ADP' or 'ATP'
	If mass: retuens slope corrected by mass.

	'''
	
	# Assign default
	alpha=1
	calibration={'MgG': alpha,
				'sc': [[0],[0]],
				'slope':float('nan'),
				'r2': float('nan'),
				'predict':[[0]]
				}

	if Mgcalib is True:
		# Calibrate to MgG
		# Needs MgG as event
		# MgG concentration 1.1uM by default
		try:
			df=df.reset_index()
			MgG_idx=df.loc[df['Event']=='MgG'].index[0]
			bsln=df[adtp].iloc[(MgG_idx-6):(MgG_idx-2)].mean()
			MgG=df[adtp].iloc[(MgG_idx+2):(MgG_idx+6)].mean()
			alpha=MgGconc/(MgG-bsln)
		except Exception as e:
			print(f'MgG calibration ERROR: {e}')

	# Differenciate between ATP and ADP calib
	# At this stage, it returns the calibrated signal after
	# the first calibration
	# If no 'ADP' or 'ATP' event in place,
	# returns the non calibrated signal
	try:
		# Lock portion of the trace to calibrate
		# Start with ADTP1 etc
		evs = [k for k,v in ATP_calib.items() if adtp in k]
		if evs:
			caldf=df.loc[df['Event'].isin(evs)]
			fst_idx=np.where(df.index==caldf.index[0])[0][0]-10
			lst_idx=np.where(df.index==caldf.index[-1])[0][0]
			
			# Lock protion to calibrae
			caldf=df.iloc[fst_idx:lst_idx]

			# Now do the calibration
			# Append calibration points to dict
			sc={} # as x:y
			for p, pdf in caldf.groupby(caldf['Event']):
				pct=int(len(pdf)*0.25)
				y = pdf[adtp].iloc[pct:len(pdf)-pct].mean()
				if p in evs: x=ATP_calib[p]
				else: x=0
				sc.update({x: y})

			# Retrieve linear sc
			x=[k for k,_ in sc.items()]
			y=[v for _,v in sc.items()]

			# Get standard curve
			lm=linear_model.LinearRegression()
			X=np.array(x).reshape(-1,1)
			model=lm.fit(X,y)
			slope=lm.coef_[0]/_mass
			r2=lm.score(X,y)

			# Do the prediction
			Y=model.predict(X)
			sc=[x, Y] # stors standard curve

			X=df[adtp].to_numpy().reshape(-1,1)
			predict=model.predict(X)

			# Keep calibration for future analysis
			calibration={
				'MgG': alpha,
				'sc': sc,
				'slope':slope,
				'r2': r2,
				'predict':predict
				}

			return calibration
		else: pass
	except Exception as e:
		print(f'ATP Calib ERROR:{e}')
	
	# If any error, return the raw ATP column 
	adtp_col=[c for c in df.columns if 'ADP' in c or 'ATP' in c][0]
	return calibration


def calibrate_ATP(chambers, _saving=False):
	'''
	Requires chambers with ADP
	'''
	calib_summary=[] # to store the final 

	for A in ['ATP', 'ADP']:
		calibs=[] # List to store A(DT)P calibrations
		ap_chambers=[c for c in chambers if A in c['protocol']]
		# Get only ADP or ATP chamber
		for chamber in ap_chambers:
			print(f"Calibrating: {chamber['Filename']}")

			# Select only ATP signal and Event column
			Amp_signal=chamber['df'].loc[:,[A,'Event']].copy()

			# Note that calibrated signal is in 'predict'
			cal=sc_ATP(Amp_signal,
							adtp=A,
							Mgcalib=False,
							ATP_calib=ATP_calib)

			cal.update({'temperature':chamber['temperature'],
						'filename':chamber['Filename'],
						'adenylate':A})
			calibs.append(cal)
			# print(f"{chamber['Filename']} => {chamber['temperature']}-{A} slope: {cal['slope']} - R2: {cal['r2']}")
	
		temp_cal=[]
		for temperature in TEMPERATURES:
			temp_calib=[]
			for cal in calibs:
				if int(cal['temperature'])==temperature:
					temp_calib.append(cal['slope'])
			temp_calib=pd.DataFrame(temp_calib, index=[0]*len(temp_calib))
			temp_cal.append(
				{'temperature': temperature,
				f'{A}_slope_mean': temp_calib[0].mean(),
					f'{A}_slope_sem': temp_calib[0].sem()
				})
		temp_cal=pd.DataFrame(temp_cal).set_index('temperature')
		calib_summary.append(temp_cal)

	calib_summary=pd.concat(calib_summary, axis=1, ignore_index=False)

	calib_summary['ratio']=calib_summary['ATP_slope_mean']/calib_summary['ADP_slope_mean']
	
	if _saving is True:
		calib_summary.to_csv('ATP_calibration_summary.csv')
	
	return calib_summary


def average_state(chamber, parameter_s=['JO2'], protocol=None, _mode='time', **kwargs):
	'''
	Retrieves averages and stores in dataframe
	
	Requires:
	- chamber: df of the chamber with at least JO2 column and time index
		events are meant to be sorted prior to this
	- parameter_s: list of parameter(s) to analyse. So far only JO2, DYm and ATP are considered
	- protocol: list of titrations
	'''


	# A few checks
	if type(parameter_s) is str: parameter_s=[parameter_s]
	if protocol is None: protocol=list(set(chamber['Event'].values))

	# If passes, d averages fo the parameter(s)
	#crate empty list to create final dataframe
	favdf=[]
	for parameter in parameter_s:

		# Create the row that will store infor ation
		row=pd.DataFrame(index=[parameter], columns=protocol)

		# Select parameter only
		pdf=pd.DataFrame(chamber.loc[:,[parameter,'Event']])

		# Some may be trimmed by reox event
		# Reset index so that delta 2s
		# between each row
		_index=list(range(0,len(pdf.index)*2,2))
		pdf=pdf.sort_index().reset_index()
		pdf['time']=_index
		pdf=pdf.set_index('time')


		#Check if calibration required
		if parameter in ['ADP','ATP']:
			# Expect coef derived from standard curve_ATP
			# and ratio for ADP
			pdf[parameter]=pdf[parameter]*kwargs['ATP_coef']
			if parameter=='ADP':
				pdf[parameter]=pdf[parameter]*kwargs['ATP_ratio']


		# Create empty columns
		# This is to ease the loc when
		# selecting the good protoin of the curve
		pdf[f'{parameter}_select']=float('nan')


		for event, pedf in pdf.groupby(pdf['Event']):
			_mean=float('nan')
			if event in protocol:

				# ========================== ANALYSE JO2 ============================
				if parameter in ['JO2', 'DYm']:
					# Select upon mode
					if _mode == 'time':
						# Check that there is enough time
						# i.e. more than 2min of recording
						dt=pedf.index[-1]-pedf.index[0]
						if dt >= 120:
							pct=int(dt*0.25) #Chose 25% after and before marks
							pedf[f"{parameter}_select"]=pedf[parameter].iloc[pct:-pct]

						# if not enough time
						# switch to 'stable mode'
						#else: _mode='stable'

					if _mode == 'stable':
						# Uses the second deriviative
						# to select the part of the trace
						# that is the most stable
						
						# Do a smoothing average to exclude
						# weird noises
						pedf['ma']=pedf[parameter].rolling(window=10
									).mean()
						# get the first deriviative
						pedf['fsder']=pedf['ma']-pedf['ma'].shift()
						# delete parts with high/low fsder
						if '_exclude' in kwargs:_exclude=kwargs['_exclude']
						else:_exclude=0.1 #Arbitrary value that seems to work
						mask=(pedf['fsder']<_exclude)&(pedf['fsder']>-_exclude)
						pedf[f'{parameter}_select']=pedf[parameter].loc[mask]


					#else:print("_mode not recognised. Choose between 'time' or 'stable'")

					# Calculate mean and averages
					# Note that CCCP, maximum is taken into account
					_mean=pedf[f'{parameter}_select'].mean()
					_sd=pedf[f'{parameter}_select'].std()


					# For JO2, only take max and or min for these states
					if parameter == 'JO2':
						if event in ['CCCP', 'TMPD-Asc','FCCP']:
							_mean=pedf[f'{parameter}_select'].max()
						if event in ['Oli', 'KCN', 'Azide']:
							_mean=pedf[f'{parameter}_select'].min()

					if parameter == 'DYm':
						if 'mass' in kwargs:
							_mean=_mean/kwargs['mass']

					# Check if _sd , 10% of _mean
					_error=_sd/_mean
					if _error>=0.5:
						#print(f"/!| Large variation in {event} ; mean={_mean} sd={_sd}|!|")
						pass

				# ======================== END OF ANALYSE JO2 =======================



				# ========================== ANALYSE ATP ============================
				elif parameter in ['ATP', 'ADP']:					
					# For ATP signal, take the start and end and
					# assume linearity.

					# Check that enough tme between start and finish
					#i.e. 20sec
					if len(pedf)>12:
						i0, i1 = 6, -6
					else: i0, i1 = 0, -1
					#print(pedf)
					dv=pedf[parameter].iloc[i1:i1+4].mean()-pedf[parameter].iloc[i0:i0+4].mean()
					dt=pedf.index[i1]-pedf.index[i0]

					# Slope
					_mass=1
					if 'mass' in kwargs:_mass=float(kwargs['mass'])
					_mean= -((dv/dt)*1000000)/_mass
					#print(f"{parameter} - {event}: {_mean}")
				#======================== END OF ANALYSE ATP ========================

				# # Graph the selected part of the curve
				# if event =='S':
				# 	pedf['ma']=pedf.rolling(window=10).mean()

				# 	pedf['fsder']=pedf[parameter]-pedf[parameter].shift()
				# 	pedf['ma_fsder']=pedf['ma']-pedf['ma'].shift()

				# 	mask=(pedf['ma_fsder']<0.08)&(pedf['ma_fsder']>-0.08)
				# 	pedf['ma_select']=pedf.loc[mask,parameter]


				# 	Graph().graph([
				# 		[pedf.index,
				# 			{'y':pedf[parameter], 'type':'line', 'color':'red'},
				# 			{'y':pedf['ma'], 'color':'green', 'type':'line'},
				# 			{'y':pedf['ma_select'],'type':'line','color':'blue'}],
				# 		[pedf.index,
				# 			{'y':pedf['fsder'], 'color':'red', 'type':'line'},
				# 			{'y':pedf['ma_fsder'], 'color':'green', 'type':'line'}]])
				# 	exit()


				row.loc[parameter,event]=_mean		
		favdf.append(row)
	favdf=pd.concat(favdf)
	return favdf


def extract_csv(**kwargs):
	'''
	To be modified for each person / experiment / way of saving data etx
	Returns:
		List of chamber, dict containing at least {'df':df}
	Note that **kwargs are passed to the split chamber
	'''
	# List of chambers to process ulteriorly.
	# This is necessary, even for one experiment
	# as the average function takes ONE chamber
	chambers=[]

	# Process each csv
	for temp in TEMPERATURES:
		try:
			csvs=[f for f in os.listdir(f"{CSV_PATH}{temp}/") if '.csv' in f]
			for csv in csvs:
				print(f"Exctracting {csv}...")

				# Open the csv
				df=pd.read_csv(f"{CSV_PATH}{temp}/{csv}",
							encoding = "ISO-8859-1")


				# Spread events for both chambers
				df=sort_events(df)

				# Split chambers
				# chamber as {'chamber': ,'protocol': ,'df':}
				chamberA, chamberB = split_chambers(df, **kwargs)
				

				meta = {'temperature':temp,
						'Filename':csv}


				chamberA.update(meta)
				chamberB.update(meta)

				#Need to append chamber to a list to process per chamber
				chambers.append(chamberA)
				chambers.append(chamberB)
		except Exception as e:
			print(f"extract_csv: {e}")
	return chambers


def average_study(chambers, protocols=None, ATP_calib=None, _saving=True):
	'''
	Wrapper function to average all chambers.
	This is mainly to have a cleaner MAIN() code
	'''
	fdf=[] #final df that contains all data

	for chamber in chambers:
		print(f"Analysing {chamber['Filename']}...")


		meta=pd.DataFrame({k:v for k,v in chamber.items() if k != 'df'},
						index=[0]
						).set_index('Filename')
		# Define titration and protocol
		# from the PROTOCOL variable
		# Also add it to 
		parameters=['JO2']
		if protocols is not None:
			protocol=chamber['protocol']
			titrations=protocols[protocol]
			if protocol in ['ADP', 'ATP', 'DYm', 'ROS', 'Ca']:
				parameters.append(protocol)
		else: protocol, titrations = None, None

		# Add kwargs specifics to each protocol
		# So far only written for ATP
		kwargs={}
		if protocol in ['ADP','ATP']:
			row=ATP_calib.loc[chamber['temperature']]
			kwargs={'ATP_coef':row[f"{protocol}_slope_mean"],
					'ATP_ratio':row['ratio'],
					'mass':chamber['mass']}

		

		avdf=average_state(chamber['df'],
						parameter_s=parameters,
						protocol=titrations,
						_mode='time',
						**kwargs)


		# Add metadata to the average df
		avdf['Filename']=chamber['Filename']
		avdf['temperature']=chamber['temperature']
		avdf['protocol']=chamber['protocol']

		# Append to the final average df
		fdf.append(avdf)

	# Create final average df from list of average df
	fdf=pd.concat(fdf, axis=0)

	if _saving is True:
		fdf.to_csv('ATP_final.csv')
		print("Saved file")

	return fdf


def main():

	# Extract the chambers acordingly, deleting reoxygenation events
	# 'Weight' specific to Alice study where weight are specified in columns 
	# 	'Weight A' and 'Weight B'
	chambers=extract_csv(get_weight_col='Weight',
						delete_reox='auto',
						window=[-5, 250])


	for chamber in chambers:
		mgfree_calib(chamber, calibration=Mgfree_calib)
		
	exit()

	# First, extract information required to calibrate ATP signal
	# It expects at least calibration performed with ADP and or ATP
	# ideally, both
	if 'ATP' in PROTOCOLS:
		ATP_CALIBRATION = calibrate_ATP(chambers)
		print(ATP_CALIBRATION)


	# Now process each chamber
	# and retrieve average for each State
	avdf = average_study(chambers,
						protocols=PROTOCOLS,
						ATP_calib=ATP_CALIBRATION,
						_saving=True)

	

	df=pd.read_csv('ATP_final.csv', index_col=0
					).reset_index(
					).rename(columns={'index':'parameter'})


	for protocol, titrations in PROTOCOLS.items():
		pdf=df.loc[df['protocol']==protocol]
		aggs={t: ['mean', 'sem'] for t in titrations}
		summary=pdf.groupby(['temperature', 'parameter']).agg(aggs)
		print(summary)
		summary.to_csv(f'{protocol}_mean.csv')




if __name__ == '__main__':
	main()
