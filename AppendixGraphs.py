# python code
import re 
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.signal import find_peaks
import matplotlib.patches as patches

#Make template with appropriate shift and scale
def make_template(amplitude, shift):
    template = [x*amplitude for x in singlet]
    for j in range(shift):
            template.pop()
            template.insert(0, 0)
    return template

#Make a total template summation
def total_template(template_list):
    Guessed_template = [0] * record_length
    for template in template_list:
        for i in range(record_length):
            Guessed_template[i]+=template[i]
    return Guessed_template

def make_plot(plot_list,peaks,bool1,shift):
    #Setup
    x_vals = []
    for i in range(len(plot_list[0])):
        x_vals.append(i/refresh_rate)
    plt.figure(figsize=(12, 9))
    
    # Set the range of x-axis
    plt.ylim(-.25, 2.1)
    plt.xlim(0, 4.8)
    
    plt.axhline(y=0, color='red', linestyle='--')
    if bool1:
        plt.axvline(x=(int_left_bound+shift)/refresh_rate, color='red', linestyle='--')
        plt.axvline(x=(int_right_bound+shift)/refresh_rate, color='red', linestyle='--')

    for plot in plot_list:
        plt.plot(x_vals,plot, label=f'List {waveform}')
    for peak in peaks:
        plt.plot(peak/refresh_rate, plot_list[0][peak],"x", color="red")

    #Full Template
    #plt.plot(template, label=f'List {waveform}', color='green')
    plt.xlabel('Time (microseconds)')
    plt.ylabel('Voltage (A.u)')
    plt.show()

# Read data from file
with open('test2.txt', 'r') as file:
    data = file.read()

# Create list of trigger times, list of waveforms with their peaks in list of lists
for t in range(1):

    # Grab record length, trigger time, and make list of lists
    for k in range(1):

        # Extract the number following "Record Length"
        record_length = int(re.search(r'Record Length:\s+(\d+)', data).group(1))

        # Extract the number following "Trigger Time Stamp"
        trigger_time_stamp = re.findall(r'Trigger Time Stamp:\s+(\d+)', data)
        trigger_time_list = []
        for stamp in trigger_time_stamp:
            trigger_time_list.append(int(stamp))


        # Extract the lists of numbers between "DC offset (DAC): 0x0CCC" and "Record Length"
        matches = re.findall(r"DC offset \(DAC\): 0x0CCC\n([\d\n]+(?:(?!Record Length)[\d\n]+)*)", data)
        number_lists_neg = []
        for match in matches:
            numbers = re.findall(r"\d+", match)
            number_lists_neg.append(list(map(int, numbers)))
        
        total_waveforms = len(number_lists_neg)

        # Switch number_lists_neg to be postive
        number_lists = []
        for list in number_lists_neg:
            num_list = []
            for i in range(len(list)):
                num_list.append(list[i]*-1)
            number_lists.append(num_list)


    # Needed to smooth out the data slightly
    for k in range(1):

        # Compute the averaged numbers for each number list
        averaged_number_lists = []

        #averaged_number_lists=number_lists
        averagingTerm = 2
        for list in number_lists:
            averaged_numbers = []
            for i in range(averagingTerm, len(list) - averagingTerm):
                average = sum(list[i-averagingTerm:i+averagingTerm+1]) / (2*averagingTerm+1)
                averaged_numbers.append(average)
            averaged_number_lists.append(averaged_numbers)
            
            #Now our indexing is off as the record length is no longer the record length
            record_length = len(averaged_number_lists[0])


    # Takes average pre-peak for each list and shifts entire list by each respective average
    for k in range(1):

        # Compute the averaged pre-peak for each list
        zero_term_list = []
        zero_term = 70
        for list in averaged_number_lists:
            zero_term_list.append(sum(list[:zero_term]) / zero_term)

        #Shift each list by their respective pre-peak average
        shifted_number_lists = []
        for i in range(total_waveforms):
            shifted_list = []
            for j in range(record_length):
                shifted_list.append(averaged_number_lists[i][j]-zero_term_list[i])
            shifted_number_lists.append(shifted_list)


    # shifted left to right, to have first peak on same spot
    center_index = 90
    for i in range(total_waveforms):
        # Find peaks of correlation
        peaks, _ = find_peaks(shifted_number_lists[i], prominence=(100,None), height=200)
        for j in range(center_index-peaks[0]):
            shifted_number_lists[i].pop()
            shifted_number_lists[i].insert(0, 0)
        


    #Find all the peaks, and store their index in a list of lists
    for k in range(1):
        peak_index_list=[]
        for waveform in shifted_number_lists:
            peaks, _ = find_peaks(waveform, prominence=(100,None), height=200)
            waveform_peak = []
            for peak in peaks:
                waveform_peak.append(peak)
            peak_index_list.append(waveform_peak)

# Integration Rules
for t in range(1):
    singlet_left_bound = 16600
    singlet_right_bound = 17600

    int_left_bound = 80
    int_right_bound = 221

# Use to make histogram of waveforms with only one peak
# Go in and type True to print histogram
for t in range(1):
    solo_peak_amplitude = []  
    for i in range(total_waveforms):
        if len(peak_index_list[i])==1:
            # Find amplitude sum of first peak
            amplitude_sum = sum(shifted_number_lists[i][int_left_bound : int_right_bound])
            solo_peak_amplitude.append(amplitude_sum)  

    #Make the histogram
    if True:
        for i in range(1):
            plt.figure(figsize=(12, 9))
            plt.hist(solo_peak_amplitude, bins=500, color='blue', edgecolor='black')

            # Add vertical dashed lines at the values of 30000 and 35000
            plt.axvline(x=singlet_left_bound, color='red', linestyle='--')
            plt.axvline(x=singlet_right_bound, color='red', linestyle='--')

            # Add vertical dashed lines at the values of 34400 and 35200
            plt.axvline(x=33000, color='orange', linestyle='--')
            plt.axvline(x=34000, color='orange', linestyle='--')

            # Add vertical dashed lines at the values of 34400 and 35200
            plt.axvline(x=50000, color='green', linestyle='--')
            plt.axvline(x=51000, color='green', linestyle='--')
            # Set histogram title and labels
            plt.title('Histogram of Amplitude Sums')
            plt.xlabel('Amplitude Sum')
            plt.ylabel('Frequency')

            # Display the histogram plot
            plt.show()

# Use to make histogram of first peak of all waveforms
# Dont need the other peak lines to be correct, just singlet peaks
for t in range(1):
    first_amplitude = []  
    for i in range(total_waveforms):
        amplitude_sum = sum(shifted_number_lists[i][int_left_bound : int_right_bound])
        first_amplitude.append(amplitude_sum)
    #Make the histogram
    if False:
        for i in range(1):
            plt.figure(figsize=(8, 6))
            plt.hist(first_amplitude, bins=500, color='blue', edgecolor='black')

            # Add vertical dashed lines at the values of 30000 and 35000
            plt.axvline(x=singlet_left_bound, color='red', linestyle='--')
            plt.axvline(x=singlet_right_bound, color='red', linestyle='--')

            # Add vertical dashed lines at the values of 34400 and 35200
            plt.axvline(x=34400, color='orange', linestyle='--')
            plt.axvline(x=35200, color='orange', linestyle='--')

            # Add vertical dashed lines at the values of 34400 and 35200
            plt.axvline(x=51000, color='green', linestyle='--')
            plt.axvline(x=52500, color='green', linestyle='--')
            # Set histogram title and labels
            plt.title('Histogram of Amplitude Sums')
            plt.xlabel('Amplitude Sum')
            plt.ylabel('Frequency')

            # Display the histogram plot
            plt.show()

#Make Singlet Template
for t in range(1):
    Guessed_Singlet = [0] * record_length  # Assuming all lists have the same length
    Singlet_Index = []
    total_singlets = 0
    for i in range(total_waveforms):
        if singlet_left_bound<=first_amplitude[i]<=singlet_right_bound:
            Singlet_Index.append(i)
            total_singlets+=1
            for j in range(record_length):
                Guessed_Singlet[j]+=shifted_number_lists[i][j]

    # Initialize the scaled singlet
    singlet = []
    for i in range(record_length):
        singlet.append(Guessed_Singlet[i]/total_singlets)

#Properly scale all waveforms so that singlet has a height of 1
for t in range(1):
    waveform_scale = singlet[center_index]
    #waveform_scale = 1
    for i in range(total_waveforms):
        shifted_number_lists[i] = [x/waveform_scale for x in shifted_number_lists[i]]
    singlet = [x/waveform_scale for x in singlet]


    #Making a scale for the amplitudes
    singlet_amp_scale = sum(singlet[int_left_bound : int_right_bound])

    #Making a new list for the repetitive step
    for m in range(1):
        index_drop_list = []
        for list in peak_index_list:
            list1=[]
            for element in list:
                list1.append(element)
            index_drop_list.append(list1)
    
    #Making a list for the shifts needed for each template
    for m in range(1):


        waveform_shifts_list = []
        for list in peak_index_list:
            list1=[]
            for element in list:
                list1.append(element-center_index)
            waveform_shifts_list.append(list1)



Amplitude_lists = []
waveform_templates_list = []



#Needed for x-values on graph
refresh_rate = 250
x_values = []
for i in range(record_length):
    x_values.append(i/refresh_rate)


#Raw waveform graph for appendix
if False:
    x_values = []
    for i in range(len(number_lists[t])):
        x_values.append(i/refresh_rate)
    t=6029
    plt.figure(figsize=(12, 9))
    plt.xlim(0, 4.8) 

    plt.plot(x_values,number_lists[t], label=f'List {waveform}')
    #Full Template
    #plt.plot(template, label=f'List {waveform}', color='green')
    plt.xlabel('Time (microseconds)')
    plt.ylabel('Voltage (miliVolts)')
    plt.show()

#Average waveform graph for appendix
if False:
    t=6029
    plt.figure(figsize=(12, 9))
    plt.xlim(0, 4.8) 

    plt.plot(x_values,averaged_number_lists[t], label=f'List {waveform}')
    #Full Template
    #plt.plot(template, label=f'List {waveform}', color='green')
    plt.xlabel('Time (microseconds)')
    plt.ylabel('Voltage (miliVolts)')
    plt.show()

#Scaled waveform graph for appendix
if False:
    t=6029
    plt.figure(figsize=(12, 9))
    plt.xlim(0, 4.8) 
    plt.axhline(y=0, color='red', linestyle='--')

    plt.plot(x_values,shifted_number_lists[t], label=f'List {waveform}')
    #Full Template
    #plt.plot(template, label=f'List {waveform}', color='green')
    plt.xlabel('Time (microseconds)')
    plt.ylabel('Voltage (A.u.)')
    plt.show()





#Iterates through templates to pull out amplitudes
for i in range(total_waveforms):
    
    Current_waveform = shifted_number_lists[i]
    peaks = peak_index_list[i]
    shifts = waveform_shifts_list[i]
    waveform_scales = []
    waveform_templates = []
    
    #Gives the approximate values for each peak:
    for j in range(len(peaks)):
        peak = peaks[j]
        shift = shifts[j]
        current_template = total_template(waveform_templates)

        this_wave =[]
        for k in range(record_length):
            this_wave.append(Current_waveform[k]-current_template[k])
        
        approx_amp = this_wave[peak]
        approx_temp = make_template(approx_amp, shift)
        waveform_templates.append(approx_temp)
    


    #Iterates through backwards and tries to find the true amp of each peak
    true_amp_list = []
    for j in range(len(peaks)):
        index = len(peaks)-j-1
        peak = peaks[index]
        shift = shifts[index]
        waveform_templates[index] = make_template(0,0)
        current_template = total_template(waveform_templates)

        this_wave =[]
        for k in range(record_length):
            this_wave.append(Current_waveform[k]-current_template[k])
        
        if peaks[index]>1064:
            true_amp = this_wave[peak]
        else:
            true_amp = sum(this_wave[int_left_bound + shift: int_right_bound + shift])/singlet_amp_scale
        true_amp_list.append(true_amp)
        waveform_templates[index] = make_template(true_amp,shift)



    #Add important infomation to lists for outside this loop
    waveform_templates_list.append(waveform_templates)
    Amplitude_lists.append(true_amp_list[::-1])



#Initialize and appending to global amps and timings
for i in range(total_waveforms):
    global_amplitudes = []
    global_timings = []
    global_indices = []
    amp_list = Amplitude_lists[i]
    peak_list = peak_index_list[i]
    global_clock = trigger_time_list[i]
    for j in range(len(amp_list)):
        amp = amp_list[j]
        peak_index = peak_list[j]
        global_amplitudes.append(amp)
        global_timings.append(peak_index + global_clock)
        global_indices.append(i)




# Find difference between adjacent trigger time stamps
trigger_time_dif = []
time_scale = 4*10**(-9)
for j in range(len(global_timings)-1):
    diff = time_scale*(global_timings[j+1] - global_timings[j])
    trigger_time_dif.append(diff)

log_time=np.log10(trigger_time_dif)

#Need to drop first in these lists because the first element will be the used as the 'start of time'
#Issue with code if first waveform has more than 1 peak, but ignore for now
for t in range(1):
    global_amplitudes = global_amplitudes[1:]
    peak_index_list = peak_index_list[1:]
    waveform_templates_list = waveform_templates_list[1:]
    shifted_number_lists = shifted_number_lists[1:]
    total_waveforms = total_waveforms-1
    Amplitude_lists = Amplitude_lists[1:]
    waveform_shifts_list = waveform_shifts_list[1:]


#Shift all the indices down one
global_indices = global_indices[1:]
index_temp = []
for i in range(len(global_indices)):
    index_temp.append(global_indices[i]-1)
global_indices = index_temp



# Pull the indices of specific waveforms
lower_limit = 3.6
upper_limit = 4.5

left_limit = -7.5
right_limit = -6.5

# Graph with scatterplot with rectangle search area
if False:
    plt.scatter(log_time, global_amplitudes,s=5,marker='o',alpha=0.3)
    #rect = patches.Rectangle((left_limit, lower_limit), right_limit - left_limit, upper_limit - lower_limit,linewidth=.75, edgecolor='red', facecolor='none')
    #plt.gca().add_patch(rect)
    plt.xlabel('Delay Time (log[s])')
    plt.ylabel('Amplitude (A.u.)')
    plt.show()

#Trying to identify certain waveforms from the scatterplot
if False:
    for i in range(len(global_amplitudes)):
        amp_bool = lower_limit <= global_amplitudes[i] <= upper_limit
        time_bool = left_limit <= log_time[i] <= right_limit
        if amp_bool and time_bool:
            this_index = global_indices[i]
            waveform = shifted_number_lists[this_index]
            template = total_template(waveform_templates_list[this_index])
            
            #Setup
            plt.figure(figsize=(12, 9))
            plt.axhline(y=0, color='red', linestyle='--')
            #plt.axvline(x=int_left_bound, color='red', linestyle='--')
            #plt.axvline(x=int_right_bound, color='red', linestyle='--')

            plt.plot(x_values,waveform, label=f'List {waveform}', color='blue')
            for peak in peak_index_list[this_index]:
                plt.plot(peak/refresh_rate, waveform[peak],"x", color="red")
            for temp in waveform_templates_list[this_index]:
                plt.plot(x_values,temp, label=f'List {waveform}')
            
            #Full Template
            #plt.plot(template, label=f'List {waveform}', color='green')
            plt.xlabel('Time')
            plt.ylabel('Voltage')
            plt.show()

# Plot each waveform with their peaks marked
if False:
    for i in range(total_waveforms):
        if len(peak_index_list[i])>1:
            
            waveform = shifted_number_lists[i]
            template = total_template(waveform_templates_list[i])
            
            #Setup
            plt.figure(figsize=(12, 9))

            # Set the range of x-axis
            plt.xlim(0, 4.8)
            plt.axhline(y=0, color='red', linestyle='--')
            #plt.axvline(x=int_left_bound, color='red', linestyle='--')
            #plt.axvline(x=int_right_bound, color='red', linestyle='--')

            
            plt.plot(x_values,waveform, label=f'List {waveform}', color='blue')
            for peak in peak_index_list[i]:
                plt.plot(peak/refresh_rate, waveform[peak],"x", color="red")
            for temp in waveform_templates_list[i]:
                plt.plot(x_values,temp, label=f'List {waveform}')
            
            #Full Template
            #plt.plot(template, label=f'List {waveform}', color='green')
            plt.xlabel('Time (microseconds)')
            plt.ylabel('Voltage (A.u.)')
            plt.show()









#Raw List of Lists: number_lists
#Smoothed Data with no shift: averaged_number_lists
#Shifted down and left and right: shifted_number_lists








#Iterates through templates to pull out amplitudes
i=6028

Current_waveform = shifted_number_lists[i]
peaks = peak_index_list[i]
shifts = waveform_shifts_list[i]
waveform_scales = []
waveform_templates = []


#Approx Amps
for k in range(1):
    #first peak
    for j in range(1):
        without1 = []
        approx_amp1 = Current_waveform[peaks[0]]
        temp1 = make_template(approx_amp1,shifts[0])
        for i in range(record_length):
            without1.append(Current_waveform[i]-temp1[i])

    #second peak
    for j in range(1):
        without2 = []
        approx_amp2 = without1[peaks[1]]
        temp2 = make_template(approx_amp2,shifts[1])
        for i in range(record_length):
            without2.append(Current_waveform[i]-temp1[i]-temp2[i])

    #third peak
    for j in range(1):
        without3 = []
        approx_amp3 = without2[peaks[2]]
        temp3 = make_template(approx_amp3,shifts[2])
        for i in range(record_length):
            without3.append(Current_waveform[i]-temp1[i]-temp2[i]-temp3[i])

#Forward Direction
if True:
    #Original Waveform
    make_plot([Current_waveform],peaks,False,0)

    #Original Waveform with first template
    make_plot([Current_waveform,temp1],peaks,False,shifts[0])

    #Without1 and the second template
    make_plot([without1,temp2],peaks[1:],False,shifts[1])

    #Without2 and the third template
    make_plot([without2,temp3],peaks[2:],False,shifts[2])

    #Without3
    make_plot([without3],[],False,0)

#True Amps
if True:
    #True Last one
    for j in range(1):
        shift = shifts[2]
        true_amp3 = sum(without2[int_left_bound + shift: int_right_bound + shift])/singlet_amp_scale
        true_temp3 = make_template(true_amp3,shift)

        make_plot([without2,true_temp3],peaks[2:],True,shift)

    #True second one
    for j in range(1):
        without1True3 = []
        for i in range(record_length):
                    without1True3.append(Current_waveform[i]-temp1[i]-true_temp3[i])

        shift = shifts[1]
        true_amp2 = sum(without1True3[int_left_bound + shift: int_right_bound + shift])/singlet_amp_scale
        true_temp2 = make_template(true_amp2,shift)

        make_plot([without1True3,true_temp2],[peaks[1]],True,shift)

    for j in range(1):
        withoutTrue23 = []
        for i in range(record_length):
                    withoutTrue23.append(Current_waveform[i]-true_temp2[i]-true_temp3[i])
        
        shift = shifts[0]
        true_amp1 = sum(withoutTrue23[int_left_bound + shift: int_right_bound + shift])/singlet_amp_scale
        true_temp1 = make_template(true_amp1,shift)

        make_plot([withoutTrue23,true_temp1],[peaks[0]],True,shift)

        print('approx: ',approx_amp1,', ',approx_amp2,', ',approx_amp3)
        print('true: ',true_amp1,', ',true_amp2,', ',true_amp3 )


#Total Template
if True:
    total_temp = [0]*len(Current_waveform)
    for m in range(len(Current_waveform)):
        for temp in [true_temp1,true_temp2,true_temp3]:
            total_temp[m]+=temp[m]
    #Setup
    plt.figure(figsize=(12, 9))
    # Set the range of x-axis
    plt.xlim(0, 4.8)
    plt.axhline(y=0, color='red', linestyle='--')
    #plt.axvline(x=int_left_bound, color='red', linestyle='--')
    #plt.axvline(x=int_right_bound, color='red', linestyle='--')
    plt.plot(x_values,Current_waveform, label=f'List {waveform}')
    for peak in peaks:
        plt.plot(peak/refresh_rate, Current_waveform[peak],"x", color="red")
    plt.plot(x_values,total_temp, label=f'List {waveform}',color = 'orange')
    
    #Full Template
    #plt.plot(template, label=f'List {waveform}', color='green')
    plt.xlabel('Time (microseconds)')
    plt.ylabel('Voltage (A.u.)')
    plt.show()


#To make graphic 4.1
if False:
    #Setup
    plt.figure(figsize=(12, 9))
    # Set the range of x-axis
    plt.ylim(-.25, 2.1)
    plt.xlim(0, 4.8)
    plt.plot(x_values,Current_waveform, label=f'List {waveform}')

    
    plt.axvline(x=(int_left_bound+shift)/refresh_rate, color='red', linestyle='--')
    plt.axvline(x=(int_right_bound+shift)/refresh_rate, color='red', linestyle='--')

    colors = ['red','orange','green']
    for m in range(len(peaks)):
        peak =peaks[m]
        shift = shifts[m]
        color1 = colors[m]
        plt.plot(peak/refresh_rate, Current_waveform[peak],"x", color=color1)
        plt.axvline(x=(int_left_bound+shift)/refresh_rate, color=color1, linestyle='--')
        plt.axvline(x=(int_right_bound+shift)/refresh_rate, color=color1, linestyle='--')

    #Full Template
    #plt.plot(template, label=f'List {waveform}', color='green')
    plt.xlabel('Time (microseconds)')
    plt.ylabel('Voltage (A.u.)')
    plt.show()