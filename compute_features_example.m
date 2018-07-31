%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute features for a set of video files from LIVE-Qualcomm databse
%

% Read subjective data
data = load('.\qualcommSubjectiveData.mat');
frates = 30;
reso = [1920 1080];

% Open feature file for output
feature_file = '.\LIVE_features.csv'; 
fid_ftr = fopen(feature_file,'w+');

% Loop through all the video files in the database
for z=1:length(data.qualcommVideoData.vidNames)

    yuv_path = '.';
    full_yuv_path = sprintf('%s/%s', yuv_path, ...
                            data.qualcommVideoData.vidNames{z});

    % Compute features for each video file
    fprintf('Computing features for sequence: %s\n',full_yuv_path);
    tic
    features = compute_nrvqa_features(full_yuv_path, reso, frates);
    toc
    
    % Write features to csv file for further processing
    fprintf(fid_ftr, '%2.2f, %2.2f,%0.2f,%0.2f', ...
            data.qualcommSubjectiveData.unBiasedMOS(z), ...
            data.qualcommSubjectiveData.biasedMOS(z), reso(1)/1920, 1);
    for j=1:length(features)
        fprintf(fid_ftr, ',%0.5f', features(j));
    end
    fprintf(fid_ftr, '\n');
  
end
fclose(fid_ftr);
fprintf('All done!\n');
