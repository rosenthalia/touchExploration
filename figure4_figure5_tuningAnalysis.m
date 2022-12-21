% figure4_figure5_tuningAnalysis
% performs linear regression tuning analysis and plots results
% also uses tuned channels to compute onset and offset times of tuned
% responses and plots these
% Isabelle Rosenthal 2022

toLoadTuning = 1; % if you want to load in the prerun bootstrapped tuning results instead of rerunning the whole analysis
toLoadTiming = 1; % if you want to load in the prerun bootstrapped timing results instead of rerunning the whole analysis
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% load in preprocessed files
allData = load(['..\Data\touchExploration_preprocessedSpks.mat']);
dataN = allData.dataN;
trialClass = allData.trialClass;
trialEffector = allData.trialEffector;
trialType = allData.trialType;
clear allData

chList = {1:96};

%get class labels - effector
labelc_eff = reshape(trialEffector,[],1);
%get class labels - modality
labelc_mod = reshape(trialClass,[],1);
%get class labels - type (just the type of touch)
labelc_ty = reshape(trialType,[],1);

%remove catch trials
label_eff = labelc_eff(labelc_eff~=0);
label_mod = labelc_mod(labelc_mod~=0);
label_ty = labelc_ty(labelc_ty~=0);

% order the conditions the way you want
modLbs = [1 4 10 13 8 11 2 5 7];
conds_mod = {'FPa',  'FPf', 'BLa','BLf',...
    'VrFPa','VrFPf','TPa','TPf','obj'};
tyLbs = [1 7 5 2 4];
conds_ty = {'FP','BL','VrFP', 'TP', 'obj'};
effLbs = [1:3];
conds_eff = {'arm','finger','obj'};
Mod2Ty = [1 1 7 7 5 5 2 2 4];
Mod2Eff = [1 2 1 2 1 2 1 2 3];

% pick times and bins to use
postStimBinLen = 40;
preStimBinLen = 80;
baselineBinEnd = 50; % number of bins to include in baseline average
binWidth = 0.5; % in sec
binStarts = [1:10:111]; % every 0.5 sec
binEnds = [10:10:120]; % every 0.5 sec
numBins = numel(binStarts);
numBinsPreTouch = numBins*(preStimBinLen/(preStimBinLen+postStimBinLen));

% color map to use for touch types
tyHues = [84/255 1/255 41/255;
    216/255 122/255 182/255;
    0/255 48/255 146/255;
    139/255 37/255 154/255;
    0.6 0.6 0.6;];

rng(21); %set the seed for replicability

clear spk_PhaseMean pLM_mod_byBinBoot pLM_mod_byBin

% format neural data
% spks_phaseMean is channel x phase x trial, where each value is mean FR across bins
% also get baseline activity for every channel
for chI = 1:numel(chList{1})
    ch = chList{1}(chI);
    %get spike data
    spk = squeeze(dataN(:,ch,:,:));
    %reshape across sessions
    spk = reshape(spk,[],2);
    %now trim all ITIs to be the same length
    spk(:,1) = cellfun(@(x) (x(max(1,size(x,2)-preStimBinLen+1):end)),spk(:,1),'un',0);
    % there's one trial that's too short so nan-pad it in the ITI
    spk(:,1) = cellfun(@(x) ([nan(1, max(preStimBinLen - size(x,2),0)) x]), spk(:,1),'un',0);
    % only pull the time range we are interested in decoding
    binnedSpksc(ch,:,:) = cell2mat(spk);
    
    % get baseline FR, mean across bins in baseline region
    spk_baselineByTrc(ch,:) = nanmean(squeeze(binnedSpksc(ch,:,1:baselineBinEnd)),2);
end

binnedSpks = binnedSpksc(:,labelc_mod~=0,:);
spk_baselineByTr = spk_baselineByTrc(:,labelc_mod~=0); %only use baseline trials from non-catch trials

% get baselines
for ch = chList{1}
    % get baseline FR, mean across trials and bins in baseline region
    spk_baseline(ch) = nanmean(nanmean(squeeze(binnedSpks(ch,:,1:baselineBinEnd))));
end

if toLoadTuning == 1
    % if you're loading in former results
    load(['..\Data\LRtuning_1000Boot.mat']);
    disp('results loaded successfully')
else
    % Linear regression analysis by modality with bootstrapping
    
    % get minimum number of trials for each category
    for cl = 1:numel(modLbs)
        trCount_mod(cl) = sum(label_mod==modLbs(cl));
    end
    minTr_mod = min(trCount_mod);
    for cl = 1:numel(tyLbs)
        trCount_ty(cl) = sum(label_ty==tyLbs(cl));
    end
    minTr_ty = min(trCount_ty);

    %one hot code the labels - with some shenanigans to avoid the catch trials
    %which are coded as 0
    mod_1HC = zeros(numel(label_mod), numel(unique((label_mod))));
    modMap = [1:numel(unique((label_mod))); modLbs]; % in display order
    for tr = 1: numel(label_mod)
        mod_1HC(tr, modMap(2,:)==label_mod(tr))=1;
    end
    % append the ITI FR, duplicated to match the number of repetitions
    modBase_1HC = [mod_1HC; zeros(minTr_mod,numel(unique((label_mod))))];
    
    % now run the linear regression
    for bn = 1:numBins
        for ch = chList{1}%for each channel
            meanVals = nanmean(squeeze(binnedSpks(ch,:,binStarts(bn):binEnds(bn))),2); % mean values within phases by channel
            % append baseline FR
            meanValsBase = [meanVals; ones(minTr_mod, 1) *spk_baseline(ch)];
            mdl = fitlm(modBase_1HC,meanValsBase);
            pLM_mod_byBin(ch, bn,:) = mdl.Coefficients.pValue(2:end); % not getting offset beta
            rLM_mod_byBin(ch, bn) = mdl.Rsquared.Ordinary; % the goodness of fit?
            rAdjustedLM_mod_byBin(ch, bn) = mdl.Rsquared.Adjusted;
            wLM_mod_byBin(ch, bn,:) = mdl.Coefficients.Estimate;
        end
        % correct for multiple comparisons within an array for each test
        for te = 1:size(pLM_mod_byBin, 3) % for each variable
            pLM_mod_byBin(chList{1}, bn,te) = bonf_holm(pLM_mod_byBin(chList{1}, bn,te));
        end
        disp(['beginning bootstrap for bin ' num2str(bn) ': on iteration 1'])
        for ii = 1:1000 % run the bootstrap
            if rem(ii,100)==0
                disp(['on iteration ' num2str(ii) ' of 1000']);
            end
            clear spk_baselineBoot lbsToUse trToUseBoot
            for cl = 1:numel(modLbs)
                trMod = find(label_mod==modLbs(cl));
                trToUseBoot(cl,:) = trMod(datasample(1:70,70))'; % pick which 70 trials to use from each class
                lbsToUse(cl,:) = ones(70,1)*modLbs(cl);
            end
            trToUseBoot = trToUseBoot(:); lbsToUse = lbsToUse(:);
            %one hot code the labels
            mod_1HCBoot = zeros(numel(lbsToUse), numel(modLbs));
            modMap = [1:numel(unique((lbsToUse))); modLbs]; % in display order
            for tr = 1: numel(lbsToUse)
                mod_1HCBoot(tr, modMap(2,:)==lbsToUse(tr))=1;
            end
            % append the ITI FR, duplicated to match the number of repetitions
            modBase_1HCBoot = [mod_1HCBoot; zeros(minTr_mod,numel(unique((lbsToUse))))];
            
            for ch = chList{1} % for each channel
                % get baseline FR, mean across trials and bins in baseline region
                spk_baselineBoot(ch) = nanmean(nanmean(squeeze(binnedSpks(ch,trToUseBoot,1:baselineBinEnd))));
                meanVals = nanmean(squeeze(binnedSpks(ch,trToUseBoot(:),binStarts(bn):binEnds(bn))),2); % mean values within phases by channel
                % append baseline FR
                meanValsBase = [meanVals; ones(minTr_mod, 1) *spk_baselineBoot(ch)];
                
                mdl = fitlm(modBase_1HCBoot,meanValsBase);
                pLM_mod_byBinBoot(ii, ch, bn,:) = mdl.Coefficients.pValue(2:end); % not getting offset beta
            end
            %  correct for multiple comparisons within an array for each test
            for te = 1:size(pLM_mod_byBinBoot, 4) % for each variable
                pLM_mod_byBinBoot(ii, chList{1}, bn,te) = bonf_holm(pLM_mod_byBinBoot(ii, chList{1}, bn,te));
            end
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% plot linear regression results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Figure 2a: bar plot showing the total number of channels tuned to each condition
% over all bins [-1 to 2]
ty2Use = [1 7 5 2]; %  FP BL vBL VrFP TP
binedges = 0.5:1:6.5;
binLbs = 1:6;
disp('linear regression tuning results: ');
clear histBins_md numBins_md numTuned numTunedBoot numTunedLowerCI numTunedUpperCI
for eff = 1:2
    % get the conditions of each type for this effector
    mod2use = find(Mod2Eff==eff);
    assert(all(Mod2Ty(mod2use)==ty2Use)); % make sure the mod2Ty mapping matches the expected order
    for tyInd = 1:numel(ty2Use)
        ty = ty2Use(tyInd);
        % get modality
        mdInd = mod2use(tyInd);
        md = modLbs(mdInd);
        % first take the mean of p values across iters, then see if sig
        numTuned(eff,tyInd) =sum(any(squeeze(pLM_mod_byBin(chList{1}, 7:end,mdInd)<0.05),2));
        
        % get bootstrap values
        numTunedBoot(eff,tyInd,:) = sum(any(squeeze(pLM_mod_byBinBoot(:, chList{1}, 7:end,mdInd))<0.05,3),2);% numTuned across all iterations
        numTunedUpperCI(eff,tyInd) = prctile(numTunedBoot(eff,tyInd,:),97.5);
        numTunedLowerCI(eff,tyInd) = prctile(numTunedBoot(eff,tyInd,:),2.5);
        
        numTunedUpperCI99(eff,tyInd) = prctile(numTunedBoot(eff,tyInd,:),99.5);
        numTunedLowerCI99(eff,tyInd) = prctile(numTunedBoot(eff,tyInd,:),0.5);
        
        numTunedUpperCI999(eff,tyInd) = prctile(numTunedBoot(eff,tyInd,:),99.95);
        numTunedLowerCI999(eff,tyInd) = prctile(numTunedBoot(eff,tyInd,:),0.05);
        
        disp([conds_mod{mdInd} ' has ' num2str(numTuned(eff,tyInd))...
            ' ch [' num2str(numTunedLowerCI(eff,tyInd)) ', ' num2str(numTunedUpperCI(eff,tyInd)) ']'])
        
    end
end
f9 = figure('Position',[800 400 800 400]); hold on
bb(1)= bar(0.8:1:3.8, numTuned(1,:)', 0.35, 'FaceColor','flat');
bb(2)= bar(1.2:1:4.2, numTuned(2,:)', 0.35,'FaceColor','flat');
xlim([0.5 4.5])
xticks(1:1:4)
ylim([0 40])

% add numbers to top of bars
for tyInd = 1:numel(ty2Use)
    bb(1).CData(tyInd,:) = tyHues(tyInd,:);
    bb(2).CData(tyInd,:) = tyHues(tyInd,:);
    if numTuned(1,tyInd)~=0
        text(tyInd-0.2,numTuned(1,tyInd)'+0.5,num2str(numTuned(1,tyInd)),'vert','bottom','horiz','center');
    end
    if numTuned(2,tyInd)~=0
        text(tyInd+0.2,numTuned(2,tyInd)'+0.5,num2str(numTuned(2,tyInd)),'vert','bottom','horiz','center');
    end
end
h=gca; h.XAxis.TickLength = [0 0];
xticklabels(conds_ty(1:4))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Figure 4b: histogram showing the number of s out of the 3 seconds from [-1 to 2] that channels
% code for a specific modality
% Note: the stripes on finger were added in an image editor afterwards because it's a real pain
% to put stripes on bars in MATLAB

ty2Use = [1 7 5 2]; %  FP BL vBL vFP vAbsFP TP], same order as in modLbs
binedges = 0.5:1:6.5;
binLbs = 1:6;
clear histBins_md numBins_md histBins_mdBoot
f7 = figure('Position',[800 400 500 400]); hold on
for eff = 1:2
    % get the conditions of each type for this effector
    mod2use = find(Mod2Eff==eff);
    assert(all(Mod2Ty(mod2use)==ty2Use)); % make sure the mod2Ty mapping matches the expected order
    for tyInd = 1:numel(ty2Use)
        ty = ty2Use(tyInd);
        % get modality
        mdInd = mod2use(tyInd);
        md = modLbs(mdInd);
        chTuned =find(any(squeeze(pLM_mod_byBin( chList{1}, 7:end,mdInd)<0.05),2));
        % get the number of s each tuned channel was significant for this modality
        numBins_md{eff,tyInd} = sum(squeeze(pLM_mod_byBin(chTuned, 7:end,mdInd)<0.05),2);
        histBins_md(eff,tyInd,:) = histcounts(numBins_md{eff,tyInd}, binedges);
    end
end
bb = bar(binLbs,[squeeze(histBins_md(1,:,:))' squeeze(histBins_md(2,:,:))'], 'stacked', 'FaceColor','flat'); 
xlim([0.5 6.5])
ylim([0 55])

% add coloring
for  tyInd = 1:numel(ty2Use)
    ty = ty2Use(tyInd);
    bb(tyInd).CData = tyHues(tyInd,:);
    bb(tyInd+numel(ty2Use)).CData = tyHues(tyInd,:);
end
xlabel('# of bins tuned in the [-1 2]s range')
ylabel('# of channels')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Figure 4c, d: circle diagrams showing which channels are tuned to which neurons
% circles are sized proportionally to the number of channels tuned

%            data is a 10 element vector of:
%               |A| where A is FP
%               |A and B|
%               |B| where B is BL
%               |B and C| 
%               |C| where c is VrFP
%               |C and A|
%               |A and B and C|
%               |D and C and A and B| where D is TP
intersectDivisions = [1 2 1 2 1 2 3 4];
intersectInds = {[1],[1 2],[2],[2 3],[3],[3 1],[1 2 3],[1 2 3 4]};
intersectTitles = {'FP','FP, BL', 'BL','BL, VrFP','VrFP','VrFP,FP','FP, BL, VrFP','FP, BL, VrFP, TP'};
effNames = {'arm','finger'};
clear tunedChByBin
for eff = 1:2
    chIntersects = zeros(8,1);
    mod2use = find(Mod2Eff==eff);
    mds = modLbs(mod2use);
    assert(all(Mod2Ty(mod2use)==ty2Use)); % make sure the mod2Ty mapping matches the expected order
    % get tuned channels by bin
    tunedCh =find(any(any(squeeze(pLM_mod_byBin( chList{1}, 7:end,mod2use))<0.05,2),3));
    % for each tuned channel, get what it is tuned to and update the
    % corresponding counters
    for chI = 1:numel(tunedCh)
        ch = tunedCh(chI);
        % get what it is tuned to across those bins)
        modsTuned = any(squeeze(pLM_mod_byBin(ch,7:end,mod2use)<0.05));
        if modsTuned(1) == 1 
            if modsTuned(2) == 1 
                if modsTuned(3) == 1 
                    chIntersects(7) = chIntersects(7) + 1; 
                    if  modsTuned(4) == 1
                        chIntersects(8) = chIntersects(8) + 1; 
                    end
                else
                    chIntersects(2) = chIntersects(2) + 1; 
                end
            elseif modsTuned(3) == 1
                chIntersects(6) = chIntersects(6) + 1; 
            else
                chIntersects(1) = chIntersects(1) + 1;
            end
        elseif modsTuned(2) == 1 
            if modsTuned(3) == 1 
                chIntersects(4) = chIntersects(4) + 1;
            else
                chIntersects(3) = chIntersects(3) + 1; 
            end
        elseif modsTuned(3) == 1 
            chIntersects(5) = chIntersects(5) + 1;
        end
        
    end
    f8 = figure('Position',[800 500 1800 900], 'Color',[1 1 1]);
    maxCh = 22;
    % draw pie charts of the right color for each type of selective tuning,
    % and size it proportional to the number of channels in that category
    for pI = 1:8
        subplot(2,4,pI)
        if chIntersects(pI)~=0
            pieData = ones(intersectDivisions(pI),1);
            p = pie(pieData); axLabels = 1:2:numel(pieData)*2;
            ax = gca(); ax.Colormap = tyHues(intersectInds{pI},:);
            % get rid of percentages
            delete(ax.Children([axLabels]))
            
            % size the pie chart appropriately
            pieAxis = get(p(1), 'Parent');
            pieAxisPosition = get(pieAxis, 'Position');
            newRadius = chIntersects(pI)/maxCh;   % Change the radius of the pie chart
            
            newPieAxisPosition = pieAxisPosition .* [1 1 newRadius newRadius];
            set(pieAxis, 'Position', newPieAxisPosition);     % Set pieAxis to new position
        end
        title([intersectTitles{pI} ', n=' num2str(chIntersects(pI))])
    end
    sgtitle(effNames{eff})
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Figure 4e: venn diagrams showing how many channels are tuned to arm/finger conditions across types
% over all bins [-1 to 2]
clear tunedCh
figure;
for eff = 1:2
    mod2use = find(Mod2Eff==eff);
    mds = modLbs(mod2use);
    assert(all(Mod2Ty(mod2use)==ty2Use)); % make sure the mod2Ty mapping matches the expected order
    % get tuned channels to any condition
    tunedCh{eff} =find(any(any(squeeze(pLM_mod_byBin( chList{1}, 7:end,mod2use))<0.05,2),3));
end
numBoth = numel(intersect(tunedCh{1},tunedCh{2}));
numArmOnly = numel(tunedCh{1}) - numBoth;
numFingerOnly = numel(tunedCh{2}) - numBoth;

disp([ 'tuning over all touch types has ' num2str(numArmOnly) ' ch in arm only'])
disp(['tuning over all touch types has ' num2str(numFingerOnly) ' ch in finger only'])
disp(['tuning over all touch types has ' num2str(numBoth) ' ch tuned to both'])
% make venn diagram
utils.vennX([numArmOnly numBoth numFingerOnly],0.005); % this is just for visualization, make pretty in affinity
v1 = gcf;
title('A v F venn of ch tuned overlap');

for eff = 1:2
    mod2use = find(Mod2Eff==eff);
    mds = modLbs(mod2use);
    assert(all(Mod2Ty(mod2use)==ty2Use)); % make sure the mod2Ty mapping matches the expected order
    % for each type, get the list of channels tuned
    for tyInd = 1:numel(ty2Use)
        ty = ty2Use(tyInd);
        % get modality
        mdInd = mod2use(tyInd);
        md = modLbs(mdInd);
        tunedChByTy{eff,tyInd} =find(any(squeeze(pLM_mod_byBin( chList{1}, 7:end,mdInd))<0.05,2));
    end
end
% now make venn diagrams if there are at least channels in 2 of 3
% categories
for tyInd = 1:numel(ty2Use)
    numBoth(tyInd) = numel(intersect(tunedChByTy{1,tyInd},tunedChByTy{2,tyInd}));
    numArmOnly(tyInd) = numel(tunedChByTy{1,tyInd}) - numBoth(tyInd);
    numFingerOnly(tyInd) = numel(tunedChByTy{2,tyInd}) - numBoth(tyInd);
    disp('~~~~~~~~~~~~~~~~~~~~~~')
    disp([conds_ty{tyInd} ' has ' num2str(numArmOnly(tyInd)) ' ch in arm only'])
    disp([conds_ty{tyInd} ' has ' num2str(numFingerOnly(tyInd)) ' ch in finger only'])
    disp([conds_ty{tyInd} ' has ' num2str(numBoth(tyInd)) ' ch tuned to both'])
    if sum([numBoth(tyInd) numArmOnly(tyInd) numFingerOnly(tyInd)]>0)>1 % if at least 2 categories
        % make venn diagram
        figure
        utils.vennX([numArmOnly(tyInd) numBoth(tyInd) numFingerOnly(tyInd)],0.005); % this is just for visualization, make pretty in affinity
        v2 = gcf;
        title([conds_ty{tyInd} ': AvF venn of ch tuned overlap'])
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Onset and offset of tuning responses analysis for FP, BL
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
typesToExamine = [1 2]; %FP BL only

% timing of bins
baseStart = 1;
binWidth = 0.05;
preTouchLb = (1:1/binWidth)*binWidth;
preTouchLb = -preTouchLb(end:-1:1);
postTouchLb = (1:2/binWidth)*binWidth;
Lbs = [preTouchLb 0 postTouchLb];
binCenters = Lbs(1:end-1) + binWidth/2;

clear crossInd onsetByElec numValidTrByCh onsetByTr chTuned
if toLoadTiming == 1
    % if you're loading in former results
    load(['..\Data\LRtiming_10000Boot.mat']); % PUT CORRECT FILE HERE
    disp('results loaded successfully')
else
    for eff = 1:2
        mod2use = find(Mod2Eff==eff);
        assert(all(Mod2Ty(mod2use)==ty2Use)); % make sure the mod2Ty mapping matches the expected order
        % for each condition
        for tyI = 1:numel(typesToExamine)
            tyInd = typesToExamine(tyI);
            ty = tyLbs(tyInd);
            % get modality
            mdInd = mod2use(tyInd);
            md = modLbs(mdInd);
            % pick out all the channels tuned to the type within [-1 to 2]
            chTuned =find(any(squeeze(pLM_mod_byBin(chList{1}, 7:end,mdInd))<0.05,2));
            trThisTy = find(label_mod==md); ct = 0; chUsed{eff,tyI} = [];
            
            dataMean = mean(squeeze(mean(binnedSpks(chTuned,trThisTy,:))));
            % get the 95% distribution of the baseline across timepoints in this trial for this elec
            lowerB = prctile(dataMean(1:baselineBinEnd),2.5);
            upperB = prctile(dataMean(1:baselineBinEnd),97.5);
            midB= prctile(dataMean(1:baselineBinEnd),50);
            % get the peak firing rate within -1 to 2s
            trTy = squeeze(dataMean(61:end));
            
            %determine if max peak is negative or positive relative to
            %distance from 50% of baseline
            [peakAcPos,peakAcIndPos] = max(trTy); % earliest peak index
            [peakAcNeg,peakAcIndNeg] = min(trTy); % earliest peak index
            
            % find the closest time before peak that the channel crosses the 95%
            % baseline threshold on average across trials
            if (peakAcPos>  upperB) % make sure it actually does cross threshold
                indOver = find(trTy(1:peakAcIndPos)> upperB); % find all inds that apply
                % make sure it doesn't go back down
                xInd = peakAcIndPos; % the smallest index connected to the max ind
                for d = numel(indOver):-1:1
                    if (xInd-indOver(d))==1
                        xInd = indOver(d);
                    end
                end
                % convert index to actual time in the center of the bin
                onsetTr = binCenters( xInd);
                
                % offset is the first point where it dips below baseline
                offInd = find(trTy(peakAcIndPos:end)<=upperB,1) + peakAcIndPos-1;
                offsetTr=binCenters(offInd);
            else
                xInd = NaN;onsetTr = NaN; %if it doesn't cross, set it to NaN
                offInd = NaN; offsetTr = NaN;
            end
            
            
            crossInd{eff,tyI} = xInd;
            offsetInd{eff, tyI} = offInd;
            basePercs{eff,tyI} = [lowerB midB upperB];
            
            % compute onset for every electrode in every condition by
            % finding the median across trials.
            onsetByTy(eff,tyI) = onsetTr;
            offsetByTy(eff,tyI) = offsetTr;
            chUsed{eff,tyI} = chTuned;
            
            % want an error estimate on the onset, so bootstrap across trials
            disp(['beginning timing bootstrap for touch type ' conds_ty{ty} ': on iteration 1'])
            for iter = 1:10000
                if rem(iter,500)==0
                    disp(['on iteration ' num2str(iter) ' of 10000']);
                end
                trToUse = datasample(trThisTy,numel(trThisTy));
                dataMeanBoot = mean(squeeze(mean(binnedSpks(chTuned,trToUse,:))));
                % get the 95% distribution of the baseline across timepoints in this trial for this elec
                upperBBoot = prctile(dataMeanBoot(1:baselineBinEnd),97.5);
                % get the peak firing rate within -1 to 2s
                trTyBoot = squeeze(dataMeanBoot(61:end));
                [peakAcPosBoot,peakAcIndPosBoot] = max(trTyBoot); % earliest peak index
                
                % find the closest time before peak that the channel crosses the 95%
                % baseline threshold on average across trials
                if (peakAcPosBoot>  upperBBoot) % make sure it actually does cross threshold
                    indOver = find(trTyBoot(1:peakAcIndPosBoot)> upperBBoot); % find all inds that apply
                    % make sure it doesn't go back down
                    xIndBoot = peakAcIndPosBoot; % the smallest index connected to the max ind
                    for d = numel(indOver):-1:1
                        if (xIndBoot-indOver(d))==1
                            xIndBoot = indOver(d);
                        end
                    end
                    
                    % just the first point where it dips below baseline
                    offIndBoot = find(trTyBoot(peakAcIndPosBoot:end)<=upperBBoot,1) + peakAcIndPosBoot-1;
                    offsetTrBoot=binCenters(offIndBoot);
                    onsetTrBoot = binCenters( xIndBoot);
                else
                    xIndBoot = NaN;onsetTrBoot = NaN; %if it doesn't cross, set it to NaN
                    offIndBoot = NaN; offsetTrBoot = NaN;
                end
                crossIndBoot{eff,tyI}(iter) = xIndBoot;
                offsetIndBoot{eff,tyI}(iter) = offIndBoot;
                
                onsetByTyBoot(eff,tyI,iter) = onsetTrBoot;
                offsetByTyBoot(eff,tyI,iter) = offsetTrBoot;
            end
            % get the boot percentiles
            onsetByTy95(eff,tyI,:)=[prctile(squeeze(onsetByTyBoot(eff,tyI,:)),2.5) prctile(squeeze(onsetByTyBoot(eff,tyI,:)),97.5)];
            onsetByTyQuarts(eff,tyI,:)=[prctile(squeeze(onsetByTyBoot(eff,tyI,:)),25) prctile(squeeze(onsetByTyBoot(eff,tyI,:)),75)];
            
            offsetByTy95(eff,tyI,:)=[prctile(squeeze(offsetByTyBoot(eff,tyI,:)),2.5) prctile(squeeze(offsetByTyBoot(eff,tyI,:)),97.5)];
        end
    end
    disp('done booting')
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Figure 5a: plot the average onsets and offsets for each touch type
disp('timing analysis results: ')
effStyles = {'o', '^'};
f5 = figure('Position', [500 400 300 550], 'Color',[1 1 1]);
for eff = 1:2
    mod2use = find(Mod2Eff==eff);
    offset = (((eff-1)*2)-1)*0.1;
    for tyI = 1:numel(typesToExamine)
        tyInd = typesToExamine(tyI);
        ty = tyLbs(tyInd);
        % get modality
        mdInd = mod2use(tyInd);
        disp([conds_mod{mdInd} ' onset: ' num2str(onsetByTy(eff,tyInd))...
            's [' num2str(onsetByTy95(eff,tyI,1)) ', ' num2str(onsetByTy95(eff,tyI,2)) ']'])
        disp([conds_mod{mdInd} ' offset: ' num2str(offsetByTy(eff,tyInd))...
            's [' num2str(offsetByTy95(eff,tyI,1)) ', ' num2str(offsetByTy95(eff,tyI,2)) ']'])
        errorbar(tyI+offset, offsetByTy(eff,tyI), offsetByTy(eff,tyI)-offsetByTy95(eff,tyI,1),...
            offsetByTy95(eff,tyI,2)-offsetByTy(eff,tyI),...
            '.','MarkerFaceColor',tyHues(tyI,:),'MarkerEdgeColor',[0 0 0], 'Color',[0 0 0])
        hold on
        errorbar(tyI+offset, onsetByTy(eff,tyI), onsetByTy(eff,tyI)-onsetByTy95(eff,tyI,1),...
            onsetByTy95(eff,tyI,2)-onsetByTy(eff,tyI),...
            '.','MarkerFaceColor',tyHues(tyI,:),'MarkerEdgeColor',[0 0 0], 'Color',[0 0 0])
        plot(tyI+offset, offsetByTy(eff,tyI), effStyles{eff},'MarkerFaceColor',tyHues(tyI,:),'MarkerEdgeColor',[0 0 0])
        plot(tyI+offset, onsetByTy(eff,tyI), effStyles{eff},'MarkerFaceColor',tyHues(tyI,:),'MarkerEdgeColor',[0 0 0])
        xlim([0 3])
        % ylim([0.1 0.25])
        xticklabels({'','FP','BL',''})
        %axis square
    end
end
line([0 4],[0 0],'Color','k')
line([0 4],[1 1],'Color','k')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Figure 5b: plot the average trace across channels and trials by modality, with onsets
effStyles = {'-','--',':'};
alpha = 0.4;
for eff = 1:2
    mod2use = find(Mod2Eff==eff);
    assert(all(Mod2Ty(mod2use)==ty2Use)); % make sure the mod2Ty mapping matches the expected order
    for tyI = 1:numel(typesToExamine)
        f6 = figure('Position', [500 400 500 300], 'Color',[1 1 1]); hold on
        tyInd = typesToExamine(tyI);
        ty = tyLbs(tyInd);
        % get modality
        mdInd = mod2use(tyInd);
        md = modLbs(mdInd);
        % pick out all the channels tuned to the type within [-1 to 2]s
        chTuned =find(any(squeeze(pLM_mod_byBin( chList{1}, 7:end,mdInd))<0.05,2));
        trThisTy = find(label_mod==md);
        % fill in the onset 95% CI
        ylim([0.5 3]);
        h = ylim;
        % add line for time zero
        line([0 0],[ylim],'Color','k','linew',1)
        xCorners = [onsetByTy95(eff,tyI,1),... % onset
            onsetByTy95(eff,tyI,1),...
            onsetByTy95(eff,tyI,2),...
            onsetByTy95(eff,tyI,2)];
        yCorners = [h h(2) h(1)];
        hf=fill(xCorners,yCorners,tyHues(tyI,:), 'FaceAlpha', alpha);
        set(hf,'edgec','none');
        
        xCorners = [offsetByTy95(eff,tyI,1),... %offset
            offsetByTy95(eff,tyI,1),...
            offsetByTy95(eff,tyI,2),...
            offsetByTy95(eff,tyI,2)];
        yCorners = [h h(2) h(1)];
        hf=fill(xCorners,yCorners,tyHues(tyI,:), 'FaceAlpha', alpha);
        set(hf,'edgec','none');
        plot(binCenters, mean(squeeze(mean(binnedSpks(chTuned,trThisTy,61:end)))),...
            effStyles{eff},'Color',tyHues(tyInd,:),'linew',2)
        % add trial timings
        line([-0.5 -0.5],[ylim],'color',[0 0 0],'LineStyle','--')
        line([1 1],[ylim],'color',[0 0 0])
        % add the onset time
        line([onsetByTy(eff,tyI) onsetByTy(eff,tyI)],[ylim],'Color','k',...
            'linew',2, 'LineStyle',':')
        % add the offset time
        line([offsetByTy(eff,tyI) offsetByTy(eff,tyI)],[ylim],'Color','k',...
            'linew',2, 'LineStyle',':')            
    end
end