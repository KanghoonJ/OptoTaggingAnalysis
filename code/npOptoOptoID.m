session = '691893_2023-10-05_12-46-57';
folderPath = 'F:\npOptoRecordings\withOpto\';
mkdir([folderPath session '\kilosort\postprocessed\experiment1_Record Node 104#Neuropix-PXI-100.ProbeA-AP_recording1\quality_metrics\']);
%% load paths
folderPath = 'F:\npOptoRecordings\withOpto\';
wfDir = [folderPath, session, '\sorted\allwaveforms\'];
timestampsDir = [folderPath, session, '\sorted\timestamps\'];
laserTimesFile = [timestampsDir 'laserTimes.npy'];
savePath = [folderPath session '\figures\'];

if ~exist(savePath,"dir")
    mkdir(savePath)
end

laserTimes = readNPY(laserTimesFile);
allFiles = dir([folderPath session]);
allFiles = {allFiles.name};
optoFile = allFiles{contains(allFiles, 'opto.csv')};
events = readtable([folderPath session '\' optoFile]);
metrics = readtable([folderPath session '\kilosort\postprocessed\experiment1_Record Node 104#Neuropix-PXI-100.ProbeA-AP_recording1\quality_metrics\metrics.csv']);
templates = readNPY([wfDir 'templates.npy']);
%% prepare events ID
sites = unique(events.('site'));
powers = unique(events.('power'));
duration = unique(events.('duration'));
intervals = unique(events.('pulse_interval'));
pulseNum = unique(events.('num_pulses'));
offset = duration + intervals;
window = [0 20/1000];
respWin = 20/1000;
pThresh = 0.4;
preLen = 500;
postLen = 1000; 
%% loop through neurons
% plot all neurons

allUnits = dir(wfDir); 
allUnits = {allUnits.name};
allUnits = allUnits(contains(allUnits, 'allMeans.npy'));
IDnumber = cell2mat(cellfun(@(x) str2double(x(5:end-13)), allUnits, 'UniformOutput',false));
[IDnumber, ind] = sort(IDnumber);
allUnits = allUnits(ind)';

stepNum = 100;
myMap = [linspace(1, 1, stepNum); linspace(1, 0, stepNum); linspace(1, 0, stepNum)]';
isiV = zeros(size(allUnits));
isiThresh = 2/1000; % violation in seconds
for i = 1:length(allUnits)
    figure;
    unitID = split(allUnits{i}, '_allMeans.npy');
    unitID = unitID{1};
    currID = IDnumber(i);
    currTemplate = squeeze(templates(i,:,:));
    [~, maxChannelTemp] = sort(min(currTemplate, [], 1));
    maxChannel(i) = mean(maxChannelTemp(1:3));
    meanWF = readNPY([wfDir allUnits{i}]);
    peakChannel = min(squeeze(meanWF(3, :, :, 1)), [], 1);
    [~, peakChannel] = min(peakChannel);
    spiketimes = readNPY([timestampsDir unitID '_allSpikes.npy']);
    isiV(i) = sum(diff(spiketimes)<isiThresh)/length(spiketimes)* 100;
    metricCurr = metrics(metrics.Var1==currID,:);
    if metricCurr.firing_rate < 0.01 || metricCurr.firing_rate > 20
        continue
    end
    for currPower = 1:length(powers)
        respNumAllSites = zeros(length(sites), pulseNum);
        respLatsAllSites = zeros(length(sites), pulseNum);
        for currSite = 1:length(sites)        
            currLasersInds = events.("power") == powers(currPower) & events.("site") == sites(currSite);
            currLaserTimes = laserTimes(currLasersInds);
            respNum = zeros(length(currLaserTimes), pulseNum);
            respLats = zeros(length(currLaserTimes), pulseNum);
            for currP = 1:pulseNum
                currAlignTime = currLaserTimes + offset/1000 * (currP - 1);
                respTimesTemp = countEvents(spiketimes, currAlignTime, window);
                respLatsTemp = NaN(length(currLaserTimes), 1);
                respLatsTemp(~cellfun(@isempty, respTimesTemp)) = cell2mat(cellfun(@min, respTimesTemp, 'UniformOutput', false));
                respLats(:,currP) = respLatsTemp;
                respNum(:,currP) = cellfun(@(x) sum(x < respWin), respTimesTemp);
            end
            respNumAllSites(currSite,:) = mean(respNum~=0);
            respLatsAllSites(currSite,:) = mean(respLats,"omitmissing");
        end
        

        subplot(2, 4*length(powers), 4*(currPower-1)+1)
        controlRate = respWin*metricCurr.firing_rate * ones(1, pulseNum);
        image(1:pulseNum, [0; sites], [controlRate; respNumAllSites]*100);
        colormap(myMap)
        colorbar
        title(powers(currPower))

        % find significant ones
        respSitesInds = find(mean(respNumAllSites>=pThresh,2)>0);
        [focusP, focusInd] = max(mean(respNumAllSites(:,:),2), [], 1);
        if focusP > 0
            alignTime = laserTimes(events.("power") == powers(currPower) & events.("site") == sites(focusInd));
    
            respLats = countEvents(spiketimes, alignTime, [-preLen/1000, postLen/1000]);
            
            respLats = cellfun(@(x) 1000*x, respLats, 'UniformOutput', false);
            subplot(2, 4*length(powers), 4*(currPower-1)+ [2:4])
            plotSpikeRaster(respLats, 'PlotType','vertline');
            hold on;
            xlim([-preLen, postLen])
            for j = 1:pulseNum
                x = (j-1)*offset;
                patch([x, x+duration, x+duration, x], [0 0 length(alignTime) length(alignTime)], 'r', 'FaceAlpha', 0.25, 'EdgeColor','none');
            end
        end

    
        for currSite = 1:length(respSitesInds)
            subplot(2*length(respSitesInds), 3*length(powers), (length(respSitesInds)+currSite-1)*3*length(powers) + 3 * (currPower - 1) + 1)
            hold on;
            plot(1:pulseNum, controlRate, 'LineWidth',2,'Color','r', 'LineStyle','--')
            plot(1:pulseNum, respNumAllSites(respSitesInds(currSite), :), 'LineWidth',2,'Color','k')
            ylabel(num2str(respSitesInds(currSite)))
            ylim([0 1])
            
            if currSite == 1
                title('P(response)')
            end        
            subplot(2*length(respSitesInds), 3*length(powers), (length(respSitesInds)+currSite-1)*3*length(powers) + 3 * (currPower - 1) + 2)
            hold on;
            plot(1:pulseNum, respLatsAllSites(respSitesInds(currSite), :), 'LineWidth',2,'Color','k')
            if currSite == 1
                title('Response latency')
            end               
            ylim([0 20/1000])
            subplot(2*length(respSitesInds), 3*length(powers), (length(respSitesInds)+currSite-1)*3*length(powers) + 3 * (currPower - 1) + 3)
            hold on;
            plot(meanWF(3,:,peakChannel), 'LineWidth',3,'Color',[0.4 0.4 0.4])
            plot(meanWF(3+respSitesInds(currSite),:,peakChannel), 'LineWidth',1,'Color','b')           

            if currSite == 1
                title('Waveform')
            end               
        end
    end
    sgtitle([sprintf([unitID, ' ISI violation %0.2f'], isiV(i)) , '%', 'peakChannel' num2str(maxChannel(i))])  
    screen = get(0,'Screensize');
    screen(4) = screen(4) - 100;
    set(gcf, 'Position', screen)
    saveFigurePDF(gcf,[savePath unitID '_optoTagging.pdf'])
    close all

end
allFiles = dir(savePath);
allFiles = {allFiles([allFiles.bytes]>0).name}';
allFiles = strcat(savePath, allFiles);
append_pdfs([savePath session 'CombinedOpto.pdf'],allFiles{:});
%% loop through neurons
% no plot

allUnits = dir(wfDir); 
allUnits = {allUnits.name};
allUnits = allUnits(contains(allUnits, 'allMeans.npy'));
IDnumber = cell2mat(cellfun(@(x) str2double(x(5:end-13)), allUnits, 'UniformOutput',false));
[IDnumber, ind] = sort(IDnumber);
allUnits = allUnits(ind)';

stepNum = 100;
myMap = [linspace(1, 1, stepNum); linspace(1, 0, stepNum); linspace(1, 0, stepNum)]';
isiV = zeros(size(allUnits));
pMax = zeros(size(allUnits));
pMean = zeros(size(allUnits));
latMin = zeros(size(allUnits));
fr = zeros(size(allUnits));
presence = zeros(size(allUnits));
maxChannel = zeros(size(allUnits));
waveforms = zeros(length(allUnits), 210);
isiThresh = 2/1000; % violation in seconds
isiMet = zeros(size(allUnits));
for i = 1:length(allUnits)
    unitID = split(allUnits{i}, '_allMeans.npy');
    unitID = unitID{1};
    currID = str2double(unitID(5:end));
    currTemplate = squeeze(templates(i,:,:));
    [~, maxChannelTemp] = sort(min(currTemplate, [], 1));
    maxChannel(i) = mean(maxChannelTemp(1:3));
    meanWF = readNPY([wfDir allUnits{i}]);
    peakChannel = min(squeeze(meanWF(3, :, :, 1)), [], 1);
    [~, peakChannel] = sort(peakChannel);
    waveforms(i,:) = mean(squeeze(meanWF(1, :, peakChannel(1), 1)), 1);
    spiketimes = readNPY([timestampsDir unitID '_allSpikes.npy']);
    isiV(i) = sum(diff(spiketimes)<isiThresh)/length(spiketimes)* 100;
    metricCurr = metrics(metrics.Var1==currID,:);
    if metricCurr.firing_rate < 0.01 || metricCurr.firing_rate > 20
        continue
    end
    isiMet(i) = metricCurr.isi_violations_ratio;
    for currPower = 1:length(powers)
        respNumAllSites = zeros(length(sites), pulseNum);
        respLatsAllSites = zeros(length(sites), pulseNum);
        for currSite = 1:length(sites)        
            currLasersInds = events.("power") == powers(currPower) & events.("site") == sites(currSite);
            currLaserTimes = laserTimes(currLasersInds);
            respNum = zeros(length(currLaserTimes), pulseNum);
            respLats = zeros(length(currLaserTimes), pulseNum);
            for currP = 1:pulseNum
                currAlignTime = currLaserTimes + offset/1000 * (currP - 1);
                respTimesTemp = countEvents(spiketimes, currAlignTime, window);
                respLatsTemp = NaN(length(currLaserTimes), 1);
                respLatsTemp(~cellfun(@isempty, respTimesTemp)) = cell2mat(cellfun(@min, respTimesTemp, 'UniformOutput', false));
                respLats(:,currP) = respLatsTemp;
                respNum(:,currP) = cellfun(@(x) sum(x < respWin), respTimesTemp);
            end
            respNumAllSites(currSite,:) = mean(respNum~=0);
            respLatsAllSites(currSite,:) = mean(respLats,"omitmissing");
        end
        
        controlRate = respWin*metricCurr.firing_rate * ones(1, pulseNum);
        % find significant ones
        respSitesInds = find(mean(respNumAllSites>=pThresh,2)>0);
        [focusP, focusInd] = max(mean(respNumAllSites(:,:),2), [], 1);
        pMax(i) = focusP;
        pMean(i) = controlRate(1);
        fr(i) = metricCurr.firing_rate;
        presence(i) = metricCurr.presence_ratio;
        if focusP > 0
            alignTime = laserTimes(events.("power") == powers(currPower) & events.("site") == sites(focusInd));
            latMin(i) = min(respLatsAllSites(focusInd,:));
        end

        

    end
end
%% opto tagging
isiT = 5; % in percentile
optoInd = find(pMax > pMean+0.5); 
   %%
optoInd = find(maxChannel>=6 & maxChannel<=49 & fr<10 & fr>0.1 & isiMet<=0.5 & presence>0.8);
%% save files
allFiles = dir(savePath);
allFiles = {allFiles([allFiles.bytes]>0).name}';
allFiles = strcat(savePath, allFiles);
optoFiles = cell(length(optoInd),1);
allOptoUnits = allUnits(optoInd);   
for j = 1:length(allOptoUnits)
    currUnit = allOptoUnits{j};
    unitID = split(currUnit, '_allMeans.npy');
    unitID = unitID{1};
    currFileInd = contains(allFiles, [unitID '_']);
    optoFiles(j) = allFiles(currFileInd);
end
append_pdfs([savePath session 'CombinedOptoIDed.pdf'],optoFiles{:});
%% plot waveforms
maxChannelOpto = maxChannel(optoInd);
waveformsOpto = waveforms(optoInd, :);
baseline = mean(waveformsOpto(:, 1:90), 2);
% realign
pre = 1/1000; % in s
post = 2/1000; 
sf = 30000;
len = (pre+post) * sf+1;

waveformsAligned = zeros(length(optoInd), len);
for i = 1:length(optoInd)
    [peak, peakInd] = min(waveformsOpto(i,:));
    waveformsAligned(i,:) = waveformsOpto(i,(peakInd-pre*sf):(peakInd+post*sf)) - baseline(1);
    waveformsAligned(i,:) = waveformsAligned(i,:)/min(waveformsAligned(i,:));
end
% %%
% figure2;
% hold on;
% for i = 1:length(optoInd)
%     plot(1:len, waveformsAligned(i,:) + maxChannelOpto(i), 'k');
% end
% %%
[~, sortedInd] = sort(maxChannelOpto);
[coeff,score,latent, ~, explained, mu] = pca(zscore(waveformsAligned, [], 1));
numCat = 2;
indAll = {};
dis = {};
for a = 1:10
    [indAll{a}, ~, dis{a}] = kmeans([score(:, 1:5)], numCat);
end
[~,optiInds] = min(cellfun(@mean, dis));
ind = indAll{optiInds};
figure2;
plot(mean(waveformsAligned(ind == 1, :)))
hold on
plot(mean(waveformsAligned(ind == 2, :)))
figure2;
edges = linspace(min(maxChannelOpto)-0.01, max(maxChannelOpto)+0.01, 10);
histogram(maxChannelOpto(ind==1), edges, 'Normalization', 'probability')
hold on;
histogram(maxChannelOpto(ind==2), edges, 'Normalization', 'probability')

figure2;
hold on;
scatter(score(ind==1, 1), score(ind==1, 2))
scatter(score(ind==2, 1), score(ind==2, 2))

edges = linspace(min(score(:,1))-0.01, max(score(:,1))+0.01, 20);
figure2;
histogram(score(ind==1,1), edges)
hold on;
histogram(score(ind==2,1), edges)
%%