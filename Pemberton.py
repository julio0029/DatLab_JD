#!/usr/bin/env python3
#-*- coding: utf-8 -*-
import os
current_path = f"{os.path.dirname(os.path.abspath(__file__))}/"


'''
Under MIT Licence.

Apologie in advance for the lack of documentation.

This script has been written for the prupose to analyse respirometry and fluorumetry data
for the "Pemberton study", done in Feb 2019 by A.Prof. Tony Hickey and Dr. Jules Devaux.

While some obvious optimisation could be applied, this script can be used 'as is'.

This requires csv files extracted from DatLab with oxygen tension, respiration, temperature and Amp (fluorimetry) parameters.
Here, it also requires a repository containing metadata associated with each csv such as acclimation temperature,
fish number, tank number etc.

Each csv is being split between two 'chambers' (only one if only one chamber)
each chamber contains its own information and then processed individually.
each of the csv can be seen as a tradeoff between data prcessing and data anlysis where ADP
is the key component and ATP is not.
It may be relevant in this case to estimate membrane potential as it is. 

'''
#=================================== PARAMETERS ===========================================

# Repository:
repositoryName = 'rep.csv'
repository_path = f"{current_path}{repositoryName}"

# Data:
filename = None # e.g."2019-02-14_21-06-20C.csv" or 'None' if analyse all file in Data folder
CSV_PATH = f"{current_path}Data/Processed/csv/"

# Experimental:
PROTOCOLS={'DYm':['TMRM', 'Heart', 'PMG', 'ADP', 'S', 'Oli', 'CCCP', 'Azide'],
		'ATP':['MgG', 'Heart', 'ATP1', 'ATP2', 'PMGS', 'Bleb', 'Ouab','ADP1', 'ADP2', 'CCCP', 'Oli']}

TEMPERATURES=[20, 25, 30, 35] #Celcius

ATP_calib = {'ATP1':1.25, #mM
			'ATP2':2.5,
			'ADP1':1.25,
			'ADP2':2.5}

chamber_V = 2 #ml

# Graph:
# Will only graph one file precised in 'filename' above
graphing = False
to_plot = ['B_O2', 'B_JO2', 'B_ATP'] # 
x_range = None # Can choose 'start' or as format: (start) / (start,end)

# Constants:
F = 96485.33212 #C mol−1
R = 8.314472 #J.K-1
Z = 1 #Valence of TMRM / safranine etc
#=========================================================================================



#////////////////////////////////////// SCRIPT \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn import linear_model
import itertools



def del_reox(df, o2col='O2', jo2col= 'JO2', _mode='auto', window=[-5,300]):
	'''
	Delete reoxygenation if required
	Please precise _mode
	By default, 'auto' will take the [O2] as ref and delete any
		[O2] going up.
		It will also delete portions were JO2 is extreme
		and this may be rpeceised in 'window' variable as [min, max].
	If _mode is 'reox':
		Expects 'reox' Events, which will be deleted until next event.
	'''
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
	Takes the yet unprocessed df (i.e. exported csv from DatLab)
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



def split_chambers(df, rename_cols=True, delete_reox=None, _index='time', **kwargs):

	"""
	Does what it means...
	Can be used for any csv extracted from DatLab.
	Does not rely on repo or other thing specific to a protocol
	YET, needs to manualy specify protocol options...
	This will need to be upgraded in future versions
	"""

	chambers=[]

	for C in ['A', 'B']:
		time_col=[c for c in df.columns if 'Time [' in c][0]
		cols = [time_col] + [c for c in df.columns if f"{C}:" in c]
		cdf=df.loc[:,cols]

		#- define columns
		o2col=[c for c in cols if 'concentration' in c][0]
		jo2col=[c for c in cols if any(w in c for w in ['per mass'])][0]
		fluo=[c for c in cols if 'Amp' in c or 'pX' in c][0]
		event_col=[c for c in cols if all([f"{C}:_Event" in c, "text" not in c])][0]
		event_text_col=[c for c in cols if all([f"{C}:_Event" in c, "text" in c])][0]
		

		# Define the protocol
		# This on the basis that
		# DYm was calibrated and in uM or nM
		# ATP in raw voltage
		for c in cols:
			if "Amp" in c or "pX" in c:
				if 'raw' in c: protocol = 'ATP'
				else: protocol = 'DYm'


		if protocol != None:
			titrations=PROTOCOLS[protocol]
		else: titrations=None

		# If temperature in columns:
		try:
			temp_col=[c for c in cols if 'Block temp' in c][0]
			temp=df[temp_col].mean()
		except: temp=float('nan')



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


		#-- delete reox
		if delete_reox != None:
			if 'window' in kwargs:
				window=kwargs['window']
			else:window=[-5,300]
			cdf=del_reox(cdf,
							_mode=delete_reox,
							window=window)



		chamber = {
			'chamber': C,
			'protocol': protocol,
			'titrations':titrations,
			'temperature':temp,
			'df': cdf
			}
		chambers.append(chamber)

	return chambers[0], chambers[1]



def extract_csv(**kwargs):
	'''
	To be modified for each person / experiment / way of saving data etx
	Returns:
		List of chamber, dict containing at least {'df':df}
	kwargs are passed to the split_chamber() function (and subsequent)
	'''


	# Load repository to retrieve
	# masses, acclimation and tank
	repo=pd.read_csv(repository_path)


	# List of chambers to process ulteriurly.
	# This is necessary, even for one experiment
	# as the average function takes ONE chamber
	chambers=[]

	# Process each csv
	csvs=[f for f in os.listdir(CSV_PATH) if '.csv' in f]
	for csv in csvs:

		print(f"Exctracting {csv}...")

		# Open the csv
		df=pd.read_csv(f"{CSV_PATH}{csv}",
					encoding = "ISO-8859-1")


		# Spread events for both chambers
		df=sort_events(df)

		
		# Split chambers
		# chamber as {'chamber': ,'protocol': ,'df':}
		chamberA, chamberB = split_chambers(df, **kwargs)
		
		# Assigne meta to chamber dictionary
		# This is specific to Pemberton
		mask_A=(repo['Filename']==csv)
		mask_B=(repo['Filename']==csv)
		# Some files have two individuals
		# So select the right row
		if len(csv)>25:
			mask_A=((repo['Filename']==csv)&(repo['Fish_Nb']==int(csv[-13:-11])))
			mask_B=((repo['Filename']==csv)&(repo['Fish_Nb']==int(csv[-10:-8])))


		meta_A=repo.loc[mask_A].iloc[0].to_dict()
		meta_B=repo.loc[mask_B].iloc[0].to_dict()

		meta_A['temperature']=csv[-7:-5]
		meta_B['temperature']=csv[-7:-5]

		chamberA.update(meta_A)
		chamberB.update(meta_B)

		chamberA['mass']=chamberA["A_mass"]
		chamberB['mass']=chamberA["B_mass"]

		#Need to append chamber to a list to process per chamber
		chambers.append(chamberA)
		chambers.append(chamberB)

	return chambers



def sc_ATP(df, adtp_col=None, adtp='ATP', ATP_calib=None, _mass=1, Mgcalib=True, MgGconc=1.1):
	'''
	Calibrate the ATP signal
	
	Requires df with at least 'Event' and 'ADP' or 'ATP'
	If mass: retuens slope corrected by mass.

	'''
	
	# Assign default
	alpha=float('nan')
	calibration={'MgG': alpha,
				'sc': [[0],[0]],
				'slope':float('nan'),
				'intercept':float('nan'),
				'r2': float('nan'),
				'predict':[[0]]
				}

	# Sort the column to calibrate
	if adtp_col==None:adtp_col=adtp

	if Mgcalib is True:
		# Calibrate to MgG
		# Needs MgG as event
		# MgG concentration 1.1uM by default
		try:
			MgG_idx=np.where(df.index==df.loc[df['Event']=='MgG'].index[0])[0][0]
			bsln=df[adtp_col].iloc[(MgG_idx-6):(MgG_idx-2)].mean()
			MgG=df[adtp_col].iloc[(MgG_idx+2):(MgG_idx+6)].mean()
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
			fst_idx=np.where(df.index==caldf.index[0])[0][0]-20
			lst_idx=np.where(df.index==caldf.index[-1])[0][0]
			
			# Lock protion to calibrae
			caldf=df.iloc[fst_idx:lst_idx]

			# Now do the calibration
			# Append calibration points to dict
			sc={} # as x:y
			for p, pdf in caldf.groupby(caldf['Event']):
				pct=int(len(pdf)*0.25)
				y = pdf[adtp_col].iloc[pct:len(pdf)-pct].mean()
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
			r2=lm.score(X,y)

			# This is for ADP calib.
			# Seems to pick up wrong 0
			# so only do 2 points cal
			if r2<=0.9:
				x, y = x[:-1], y[:-1]
				X=np.array(x).reshape(-1,1)
				#lm=linear_model.LinearRegression()
				model=lm.fit(X,y)
				r2=lm.score(X,y)

			sc_raw=[x,y]# store to check sc
			slope=lm.coef_[0]

			# Do the prediction
			Y=model.predict(X)
			sc=[x, Y] # stors standard curve

			X=df[adtp_col].to_numpy().reshape(-1,1)
			predict=model.predict(X)

			# Keep calibration for future analysis
			calibration={
				'MgG': alpha,
				'sc_raw':sc_raw,
				'sc': sc,
				'slope':slope,
				'intercept':lm.intercept_,
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



def average_states(chamber_df, parameter_s=['JO2'], titrations=None, _mode='time', _mass=1, **kwargs):
	'''
	Retrieves averages and stores in dataframe
	
	Requires:
	- chamber: df of the chamber with at least JO2 column and time index
		events are meant to be sorted prior to this
	- _mode: 'stable' 
	'''


	# A few checks
	if type(parameter_s) is str: parameter_s=[parameter_s]
	if titrations is None: titrations=list(set(chamber_df['Event'].values))
	if 'mass' in kwargs: _mass=kwargs['mass']

	#crate empty list to create final dataframe
	favdf=[]
	for parameter in parameter_s:

		# Create the row that will store infor ation
		row=pd.DataFrame(index=[parameter], columns=titrations)

		# Select parameter only
		pdf=pd.DataFrame(chamber_df.loc[:,[parameter,'Event']]).reset_index()

		for event, pedf in pdf.groupby(pdf['Event']):
			_mean=float('nan')
			if event in titrations:

				# First sort time in case of reoxygenation deletions
				# and smooth signal
				pedf=pedf.sort_index().rolling(window=5, min_periods=1).mean()
				dt=pedf['time'].iloc[-1]-pedf['time'].iloc[0]


				# Select upon mode
				if (_mode == 'time')&(dt >= 120):
					# Check that there is enough time
					# i.e. more than 2min of recording
					select=pedf[parameter].iloc[30:-10]

				# if not enough time
				# switch to 'stable mode'
				else: _mode='stable'
				# go within each parameter for
				# Specificity

				# ========================== ANALYSE JO2 ============================
				if parameter in ['JO2', 'DYm']:

					if _mode == 'stable':
						# Uses the second deriviative
						# to select the part of the trace
						# that is the most stable
		
						# get the first deriviative
						pedf['fsder']=pedf[parameter].diff()/pedf['time'].diff()
						# delete parts with high/low fsder
						if '_exclude' in kwargs:_exclude=kwargs['_exclude']
						else:_exclude=0.1 #Arbitrary value that seems to work
						mask=(pedf['fsder']<_exclude)&(pedf['fsder']>-_exclude)
						select=pedf[parameter].loc[mask]


					# Calculate mean and averages
					# Note that CCCP, maximum is taken into account
					_mean=select.mean()
					_sd=select.std()


					# For JO2, only take max and or min for these states
					if parameter == 'JO2':
						if event in ['CCCP', 'TMPD-Asc','FCCP']:
							_mean=select.max()
						if event in ['Oli', 'KCN', 'Azide']:
							_mean=select.min()


					# Check if _sd , 10% of _mean
					_error=_sd/_mean
					if _error>=0.5:
						#print(f"/!| Large variation in {event} ; mean={_mean} sd={_sd}|!|")
						pass

				# ======================== END OF ANALYSE JO2 =======================


				# ========================== ANALYSE ATP ============================
				elif parameter in ['ATP', 'ADP']:

					'''
					Expects to have ATP signal calibrated already
					Note that at this stage, [ATP] is affected by jumps
					in fluo due to substrate titrations, however
					this is cancelled in rate calculation...
					'''

					# discard 20% start and 20% end
					idx=int(len(pedf)*0.2)
					pedf=pedf.iloc[idx:len(pedf)-idx]

					smooth=pedf[parameter].rolling(window=4, min_periods=1).mean()

					# try:
					# 	X=np.array(smooth).reshape(-1,1)
					# 	y=pedf['time']
					# 	lm=linear_model.LinearRegression().fit(X,y)
					# 	r2=lm.score(X,y)
					# 	slope=lm.coef_[0]
					# except: slope=(smooth.diff()/pedf['time'].diff()).mean()
					slope=(smooth.diff()/pedf['time'].diff()).mean()
					_mean=slope/_mass


				row.loc[parameter,event]=_mean		
		favdf.append(row)
	favdf=pd.concat(favdf)
	return favdf

def do_calculations(avdf, protocol):

	if protocol=='ATP':

	# ============== Do calculations ==================
		# On the fly since every protocol is different
		JO2=(avdf['Parameter']=='JO2')
		ATP=(avdf['Parameter']=='ATP')

		# --- delete Oli background from the rest?

		# --- net ATP
		avdf.loc[ATP,'net_ATP']=avdf.loc[ATP,'ADP2']-avdf.loc[ATP,'ATP2']


		# --- Rels and PO ratio
		for c in PROTOCOLS['ATP']:
			avdf.loc[ATP,f'{c}_rel_ETS']=avdf.loc[ATP,c]/avdf.loc[ATP,'Heart']
			avdf.loc[JO2,f'{c}_rel_ETS']=avdf.loc[JO2,c]/avdf.loc[JO2,'CCCP']
			# PO
			avdf.loc[ATP,f"POabs_{c}"]=2*(avdf.loc[ATP,c].values/avdf.loc[JO2,c].values)
			avdf.loc[ATP,f"POrel_{c}"]=2*(avdf.loc[ATP,f'{c}_rel_ETS'].values/avdf.loc[JO2,f'{c}_rel_ETS'].values)


		# --- rel_ETS
		for col in ['PMGS', 'ADP2']:
			avdf.loc[JO2,f'{col}_rel_ETS']=avdf.loc[JO2,col]/avdf.loc[JO2,'CCCP']


		# --- Inhibitor contribution
		avdf['Inh_contrib']=avdf['Ouab']-avdf['ATP2']
		avdf['Inh_contrib_rel_Oxphos']=avdf['Inh_contrib']/avdf['ADP2']


	elif protocol=='DYm':
		# On the fly since every protocol is different
		JO2=(avdf['Parameter']=='JO2')
		DYm=(avdf['Parameter']=='DYm')

		# CI-CII contribution
		avdf['CI_pct']=avdf['ADP']/avdf['S']
		avdf['CII_pct']=1-avdf['CI_pct']

		# RCR
		avdf.loc[JO2,'RCR']=(avdf.loc[JO2,'S']-avdf.loc[JO2,'Oli'])/avdf.loc[JO2,'S']

		# Rel-ETS and 'Work'
		for c in PROTOCOLS['DYm']:
			avdf.loc[JO2,f'{c}_rel_ETS']=avdf.loc[JO2,c]/avdf.loc[JO2,'CCCP']
			avdf.loc[JO2,f'{c}_work']=avdf.loc[JO2,c].values/avdf.loc[DYm,c].values


	return avdf


def average_study(chambers, protocol, _saving=True):
	'''

	'''
	fdf=[]						# Final df
	ATP_CALIBRATIONS=[]			# Store the ATP calibrations


	for chamber in chambers:

		print(f"Analysing {chamber['Filename']}...")

		meta=pd.DataFrame({k:v for k,v in chamber.items() if k not in ['df', 'titrations']},
						index=[0])

		if protocol=='ATP':
			# ---- Perform calibration of the Amp or pX signal ----
			# First select the two columns required
			tcdf=chamber['df'].loc[:,['ATP','Event']]
			cal_df=pd.DataFrame({
					'Filename': chamber['Filename'],
					'Acclimation': chamber['Acclimation'],
					'temperature': chamber['temperature']},
					index=[0])
			for adtp in ['ATP', 'ADP']:
				calibration = sc_ATP(tcdf,
									ATP_calib=ATP_calib,
									adtp_col='ATP',
									adtp=adtp,
									Mgcalib=False)

				cal_df[f'{adtp}_slope']=calibration['slope']
				cal_df[f'{adtp}_intercept']=calibration['intercept']
				cal_df[f'{adtp}_R2']=calibration['r2']
			_ratio=cal_df['ATP_slope'][0]/cal_df['ADP_slope'][0]
			if any([_ratio>10, _ratio<-10]):_ratio=0 #0 to avoid pb with calibration afterward
			cal_df['ratio']=_ratio
			ATP_CALIBRATIONS.append(cal_df)
			# ------------------ End calibration -------------------


			# ---- Calibrate the fluo signal using ATP calibration
			# based on [ATP] = 
			#chamber['df']['ATP']=(chamber['df']['ATP']-1)/(cal_df['ratio'][0]-1)
			chamber['df']['ATP']=chamber['df']['ATP']*cal_df['ATP_slope'][0]

			#deriv=chamber['df']['ATP'].rolling(window=5).mean().diff()/chamber['mass']




		elif protocol=='DYm':

			# --- retrieve max DYm and substract to rest
			# this to get [TMRM] in mt
			_start=chamber['df'].loc[chamber['df']['Event']=='Heart','DYm'].iloc[10:15]
			_amount=_start.mean()

			# --- estimate drift
			_uncpld=chamber['df'].loc[chamber['df']['Event']=='CCCP','DYm'].iloc[-15:-10]
			try:
				_d=(_amount-_uncpld.mean())/(_uncpld.index[0]-_start.index[0])
				_drift=_d*(chamber['df'].index-_start.index[0])

			except:_drift=0

			chamber['df'].loc[:,'DYm']=(_amount-chamber['df'].loc[:,'DYm']+_drift)/chamber['mass']


		# ============= Average per event
		avdf=average_states(chamber['df'],
							parameter_s=['JO2', protocol],
							titrations=PROTOCOLS[protocol],
							_mass=chamber['mass']
							).reset_index(
							).rename(columns={'index':'Parameter'})
			


		# ============== Do calculations ==================
		avdf=do_calculations(avdf, protocol=protocol)


		# ---- Append meta
		# Duplicate meta rows if avdf has >1 row
		# i.e. if fluo with respiration
		try:meta=meta.append([meta]*(len(avdf)-1)).reset_index().drop(columns=['index'])
		except:pass
		# Concat meta to average df
		avdf=pd.concat([meta, avdf], axis=1, ignore_index=False)
				
		fdf.append(avdf)


	# ==== Sort calibrations.
	if protocol=='ATP':
		ATP_CALIBRATIONS=pd.concat(ATP_CALIBRATIONS, axis=0).reset_index().drop(columns=['index'])
		ATP_CALIBRATIONS.to_csv('ATP_CALIBRATIONS.csv')
		print(ATP_CALIBRATIONS)


	# ==== Create summary df
	fdf=pd.concat(fdf, axis=0, ignore_index=True)
	if _saving:
		fdf.to_csv(f'{protocol}_summary.csv')

	return fdf




def do_stats(pdf, group_s=['Parameter', 'Acclimation', 'temperature'], protocol=None, titrations=None, _saving=True):
	'''
	Averages per state and per group if any.
	group_s can be multiple, as a list going top to bottom 
	'''

	#pdf=pdf.astype('float64', errors='ignore')

	# # Groupby manually
	print(f"Averaging {protocol} protocol...")
	aggs={t: ['mean', 'sem'] for t in titrations}
	summary=pdf.groupby(group_s).agg(aggs)
	summary.to_csv(f'{protocol}_mean.csv')
	print(f"Saved {protocol}_mean.csv")


	# -------- Pingouin way
	print(f"Performing three-way ANOVA for {protocol} protocol...")
	import pingouin as pg
	Stats={}
	for parameter, ppdf in pdf.groupby('Parameter'):

		#---- Convert for three way
		ppdf=ppdf.dropna(axis=1, how='all')
		tempdf=[]
		non_titr_cols=[c for c in ppdf.columns if c not in titrations]
		for c in titrations:
			try:
				df_t=pd.DataFrame(ppdf.loc[:,[c]+non_titr_cols])
				df_t['State']=c
				df_t.rename(columns={c:parameter}, inplace=True)
				tempdf.append(df_t)
			except Exception as e:print(e)
		ldf=pd.concat(tempdf, axis=0, ignore_index=True)
		#---- end df converted for three_way


		# === Do ANOVA ===
		betweens=['Acclimation', 'temperature', 'State']

		aov=pg.anova(ldf, dv=parameter,
					between=betweens,
					detailed=True)

		# Add Info of what was compared to ANOVA result
		gdf=pd.DataFrame({
			'Index':['Protocol', 'Parameter', float('nan')],
			'Value':[protocol, parameter, float('nan')]
			})
		aov=gdf.append(aov, ignore_index=True)


		# === Do Post-hoc ===
		phoc=[]
		groups=[
			{'groups': ['Acclimation', 'temperature'], 'between': 'State'},
			{'groups': ['Acclimation', 'State'], 'between': 'temperature'},
			{'groups': ['temperature', 'State'], 'between': 'Acclimation'},]
		for g in groups:
			try:
				for group, gdf in ldf.groupby(g['groups']):
					# Do the test
					ph=pg.pairwise_tukey(data=gdf, dv=parameter, between=g['between'])
					# Append groups and info to test
					_g=pd.DataFrame(list(group)+[parameter],index=g['groups']+['Parameter']).T
					ph=pd.concat([_g,ph], axis=1, ignore_index=False)
					ph[g['groups']+['Parameter']]=ph[g['groups']+['Parameter']].fillna(method='ffill')
					phoc.append(ph)
			except Exception as e:print(f"PostHoc ERROR: {e}")

		Stats.update({parameter:{'ANOVA':aov,'POSTHOC':phoc}})
	


	# === For each protocol, save Stats results into excel
	# One sheet for ANOVA results
	# One sheet for PostHoc
	fileName=f'{protocol}_Stats.xlsx'
	writer = pd.ExcelWriter(fileName, engine='xlsxwriter')
	for param, stats in Stats.items():
		for test in ['ANOVA', 'POSTHOC']:
			row=0
			if type(stats[test]) is not list:
				stats[test]=[stats[test]]
			for exdf in stats[test]:
				exdf.to_excel(writer, sheet_name=f"{param}_{test}", startrow=row , startcol=0)   
				row+=(len(exdf.index)+3) #3 being space
	
	if _saving is True:
		writer.save()
		print(f'Saved {fileName}')

	return Stats






if __name__ == '__main__':
	
	chambers=extract_csv()


	for protocol in ['ATP','DYm']:

		sel_chambers=[c for c in chambers if protocol in c['protocol']]
		fdf=average_study(sel_chambers, protocol=protocol)

		print(fdf)

		# ==== DO STATS
		# select columns to analyse
		fdf=pd.read_csv(f'{protocol}_summary.csv', index_col=0)
		meta_cols=[k for k,_ in chambers[0].items()]+['Parameter']
		stats_cols=[c for c in fdf.columns if c not in meta_cols]

		do_stats(fdf, protocol=protocol, titrations=stats_cols)
	






