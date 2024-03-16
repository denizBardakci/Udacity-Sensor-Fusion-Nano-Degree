clear all
clc;

%% Radar Specifications 
%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Frequency of operation = 77GHz
% Max Range = 200m
% Range Resolution = 1 m
% Max Velocity = 100 m/s
%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Radar Operational Parameters
maxRadarRange_m   = 200;           % Maximum range of the radar in meters
rangeResolution_m = 1;             % Range resolution of the radar in meters
maxVelocity_mps   = 70;            % Maximum velocity in meters per second
speedOfLight_mps  = 3 * 10^8;      % Speed of light in meters per second

%speed of light = 3e8
%% User Defined Range and Velocity of target
% *%TODO* :
% define the target's initial position and velocity. Note : Velocity
% remains contant
 
rangeOfTarget_m        = 75;       
velocityOfTarget_mps   = -20;     % Velocity of target

%% FMCW Waveform Generation

% *%TODO* :
%Design the FMCW waveform by giving the specs of each of its parameters.
% Calculate the Bandwidth (B), Chirp Time (Tchirp) and Slope (slope) of the FMCW
% chirp using the requirements above.

% Calculate the bandwidth of the chirp needed to achieve the desired range resolution.
B_sweep = speedOfLight_mps / (2 * rangeResolution_m);            

% Calculate the chirp time to ensure the radar signal covers the maximum radar range.
% The factor 5.5 ensures that the chirp duration is long enough to account for the round trip
% of the radar signal at the maximum range and a little beyond.
Tchirp = 5.5 * 2 * maxRadarRange_m / speedOfLight_mps;          

% Calculate the slope of the chirp signal, which is the rate of frequency change over time.
% The slope is critical for determining the range and velocity of targets.
slope = B_sweep / Tchirp;

%Operating carrier frequency of Radar 
fc= 77e9;             %carrier freq

                                                          
%The number of chirps in one sequence. Its ideal to have 2^ value for the ease of running the FFT
%for Doppler Estimation. 
Nd=128;                   % #of doppler cells OR #of sent periods % number of chirps

%The number of samples on each chirp. 
Nr=1024;                  %for length of time OR # of range cells

% Timestamp for running the displacement scenario for every sample on each
% chirp
t=linspace(0,Nd*Tchirp,Nr*Nd); %total time for samples


%Creating the vectors for Tx, Rx and Mix based on the total samples input.
Tx=zeros(1,length(t)); %transmitted signal
Rx=zeros(1,length(t)); %received signal
Mix = zeros(1,length(t)); %beat signal

%Similar vectors for range_covered and time delay.
r_t=zeros(1,length(t));
td=zeros(1,length(t));


%% Signal generation and Moving Target simulation
% Running the radar scenario over the time. 

for i=1:length(t)         
        
    currentTime = t(i);
    
    % *%TODO* :
    %For each time stamp update the Range of the Target for constant velocity. 
    rangeOfTarget_m = rangeOfTarget_m + (Tchirp/Nr)*velocityOfTarget_mps;
    r_t(i) = rangeOfTarget_m;
    
    % *%TODO* :
    %For each time sample we need update the transmitted and
    %received signal. 
   
    % The transmitted signal is a cosine wave modulated by a linear frequency modulated chirp.
    Tx(i) = cos (2 * pi * (fc*currentTime +  (slope*currentTime^2)/2 ) );

    % Calculate the round-trip time for the signal from radar to target and back.
    tripTime = 2 * rangeOfTarget_m / speedOfLight_mps;
    td(i) = tripTime;
    
    % The received signal is delayed by the round-trip time and modulated by the same LFM chirp.
    Rx (i) = cos (2 * pi * (fc* (currentTime - tripTime) +  (slope * (currentTime - tripTime)^2)/2 ) );
    
    % *%TODO* :
    %Now by mixing the Transmit and Receive generate the beat signal
    %This is done by element wise matrix multiplication of Transmit and
    %Receiver Signal
    Mix(i) = Tx(i) .* Rx (i);
        
end

%% RANGE MEASUREMENT


% *%TODO* :
%reshape the vector into Nr*Nd array. Nr and Nd here would also define the size of
%Range and Doppler FFT respectively.
Mix = reshape(Mix, [Nr, Nd]);

% *%TODO* :
%run the FFT on the beat signal along the range bins dimension (Nr) and
%normalize.
signal1DFFT = fft(Mix, Nr)./Nr;

% *%TODO* :
% Take the absolute value of FFT output
signal1DFFT = abs(signal1DFFT);

% *%TODO* :
% Output of FFT is double sided signal, but we are interested in only one side of the spectrum.
% Hence we throw out half of the samples.
signal1DFFT = signal1DFFT(1:Nr/2 + 1);

%plotting the range
figure ('Name','Range from First FFT')
subplot(2,1,1)

 % *%TODO* :
 % plot FFT output 
plot(signal1DFFT)
title(" 1D FFT Ranges ")
xlabel("Range (m)")
axis ([0 200 0 1]);



%% RANGE DOPPLER RESPONSE
% The 2D FFT implementation is already provided here. This will run a 2DFFT
% on the mixed signal (beat signal) output and generate a range doppler
% map.You will implement CFAR on the generated RDM


% Range Doppler Map Generation.

% The output of the 2D FFT is an image that has reponse in the range and
% doppler FFT bins. So, it is important to convert the axis from bin sizes
% to range and doppler based on their Max values.

Mix=reshape(Mix,[Nr,Nd]);

% 2D FFT using the FFT size for both dimensions.
sig_fft2 = fft2(Mix,Nr,Nd);

% Taking just one side of signal from Range dimension.
sig_fft2 = sig_fft2(1:Nr/2,1:Nd);
sig_fft2 = fftshift (sig_fft2);
RDM = abs(sig_fft2);
RDM = 10*log10(RDM) ;

%use the surf function to plot the output of 2DFFT and to show axis in both
%dimensions
doppler_axis = linspace(-100,100,Nd);
range_axis = linspace(-200,200,Nr/2)*((Nr/2)/400);
figure,surf(doppler_axis,range_axis,RDM);

%% CFAR implementation

%Slide Window through the complete Range Doppler Map

% *%TODO* :
%Select the number of Training Cells in both the dimensions.
NORangeTrainingCell   = 6;
NODopplerTrainingCell = 6;

% *%TODO* :
%Select the number of Guard Cells in both dimensions around the Cell under 
%test (CUT) for accurate estimation
NORangeGuardCells   = 3;
NODopplerGuardCells = 3;

% *%TODO* :
% offset the threshold by SNR value in dB
offset = 5;


% *%TODO* :
%Create a vector to store noise_level for each iteration on training cells
noise_level   = zeros(1,1);
sizeOfRange   = 2 * (NORangeTrainingCell + NORangeGuardCells) + 1;
sizeOfDoppler = 2 * (NODopplerTrainingCell + NODopplerGuardCells) + 1;


% *%TODO* :
%design a loop such that it slides the CUT across range doppler map by
%giving margins at the edges for Training and Guard Cells.
%For every iteration sum the signal level within all the training
%cells. To sum convert the value from logarithmic to linear using db2pow
%function. Average the summed values for all of the training
%cells used. After averaging convert it back to logarithimic using pow2db.
%Further add the offset to it to determine the threshold. Next, compare the
%signal under CUT with this threshold. If the CUT level > threshold assign
%it a value of 1, else equate it to 0.

% Use RDM[x,y] as the matrix from the output of 2D FFT for implementing
% CFAR
% Constants for CFAR calculation
numGuardCells = (2*NORangeGuardCells + 1) * (2*NODopplerGuardCells + 1);
numTrainingCells = (2*(NORangeTrainingCell + NORangeGuardCells) + 1) * ...
                   (2*(NODopplerTrainingCell + NODopplerGuardCells) + 1) - numGuardCells;

% Preallocate the CFAR output matrix
signal_cfar = zeros(Nr/2, Nd);

% Convert RDM from dB to power once to avoid repeated conversions
RDM_Power = db2pow(RDM);

% Loop through cells of the RDM, avoiding edges where the window would extend beyond the matrix bounds
for i = (1 + NORangeTrainingCell + NORangeGuardCells):(Nr/2 - NORangeTrainingCell - NORangeGuardCells)
    for j = (1 + NODopplerTrainingCell + NODopplerGuardCells):(Nd - NODopplerTrainingCell - NODopplerGuardCells)
        % Sum the power within the entire window
        sumTotalPower = sum(RDM_Power(i-(NORangeTrainingCell + NORangeGuardCells):i+(NORangeTrainingCell + NORangeGuardCells), ...
                                      j-(NODopplerTrainingCell + NODopplerGuardCells):j+(NODopplerTrainingCell + NODopplerGuardCells)), 'all');
        
        % Subtract the power of the guard cells and CUT to isolate the training cells' power
        sumGuardPower = sum(RDM_Power(i-NORangeGuardCells:i+NORangeGuardCells, ...
                                      j-NODopplerGuardCells:j+NODopplerGuardCells), 'all');
        noiseLevel = sumTotalPower - sumGuardPower;

        % Calculate the average noise level in the training cells and adjust by the offset
        avgNoiseLevel = noiseLevel / numTrainingCells;
        threshold = db2pow(pow2db(avgNoiseLevel) + offset);

        % Detect if the CUT exceeds the threshold
        if RDM_Power(i,j) > threshold
            signal_cfar(i,j) = 1;
        else
            signal_cfar(i,j) = 0;
        end
    end
end
 
% *%TODO* :
% The process above will generate a thresholded block, which is smaller 
%than the Range Doppler Map as the CUT cannot be located at the edges of
%matrix. Hence,few cells will not be thresholded. To keep the map size same
% set those values to 0. 
 
% *%TODO* :
%display the CFAR output using the Surf function like we did for Range
%Doppler Response output.

% Set the cells that were not processed to zero as they are on the edges.

% Display the CFAR output using the Surf function
figure,surf(doppler_axis,range_axis,signal_cfar);
colorbar;

 
 