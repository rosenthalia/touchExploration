% figure2a_pairwiseModalityDecoding
% LDA decoding on top PCs of multi unit firing rates, pairwise between different modalities
% then plots the decoding results as shown in Figure 2(a)
% Isabelle Rosenthal 2022

% parameters
decodeOnlyDorsal = 0;  % if you only want to use the dorsal array instead of the entire channel set
toLoad = 1; % if you want to load in a prerun model's results instead of rerunning the whole analysis
useSVD = 1; % to use SVD to perform dimensionality reduction prior to running the model (default = 1)
numSVs = 40; % number of PCs to use (default = 40)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load in preprocessed files
allData = load(['..\Data\touchExploration_preprocessedSpks.mat']);
dataN = allData.dataN;
trialClass = allData.trialClass;
trialEffector = allData.trialEffector;
trialType = allData.trialType;
if decodeOnlyDorsal % pull dorsal array channels if needed
    chList = {allData.dorsalArrayCh};
else
    chList = {1:96};
end
clear allData

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
binWidth = 0.5; % in sec
binStarts = [1:10:111]; % every 0.5 sec
binEnds = [10:10:120]; % every 0.5 sec
numBins = numel(binStarts);
    
% set name of analysis
sgt = 'touch-aligned LDA, pairwise modality';
if useSVD == 1
    sgt = [sgt ' on top ' num2str(numSVs) ' dims (SVD)'];
else
    sgt = [sgt ' on all channels'];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if toLoad == 1
    % if you're loading in former results
    load(['..\Data\40dimSVD_LDAdecoding_PairwiseMod_0.5sbins.mat']);
    disp('results loaded successfully')
else
    % otherwise run LDA decoding - pairwise by modality
    clear binnedSpks nullacc_half acc_half nullacc_half_confMat ...
        acc_half_confMat
    
    % sort out the respective classes and remap to be in correct order
    probClasses_g = nchoosek(1:numel(conds_mod),2);
    probClasses = probClasses_g;
    for mod = 1:numel(modLbs)
        probClasses(probClasses_g(:,1)==mod,1) = modLbs(mod);
        probClasses(probClasses_g(:,2)==mod,2) = modLbs(mod);
    end
    numProb = size(probClasses,1);
    
    numBinsPreTouch = numBins*(preStimBinLen/(preStimBinLen+postStimBinLen));
    
    % spks_phaseMean is channel x phase x trial, where each value is mean FR across bins
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
        binnedSpks(ch,:,:) = cell2mat(spk);
    end
    %get class labels - modality
    labelc_mod = reshape(trialClass,[],1);
    
    %remove catch trials
    label_mod = labelc_mod(labelc_mod~=0);
    
    binnedSpks_noC = binnedSpks(:,labelc_mod~=0,:);
      
    % get equal numbers of trials for each class, balanced across
    % modality
    classesUsed = unique(probClasses);
    for cl = 1:numel(classesUsed)
        trCount(cl) = sum(label_mod==classesUsed(cl));
    end
    numTrToUse = round(min(trCount)/2); %trials for train and test
    for bn = 1:numel(binStarts)
        disp(['running bin ' num2str(bn) '/' num2str(numel(binStarts))]);
        % sort binnedSpks into appropriate bins and mean
        bnspk = nanmean(binnedSpks_noC(:,:,binStarts(bn):binEnds(bn)),3);
        
        % for now exclude ch 73 on nsp 1
        x = squeeze(bnspk(chList{1},:))';
        
        %50/50 cross validation
        for rep=1:1000
            % randomly divide each class used into training and testing
            % subsections by dividing the minimum trials usable into two
            trialsTrain = []; labelsTrain = [];
            trialsTest = []; labelsTest = [];
            for cl = 1:numel(classesUsed)
                trInd = find(label_mod==classesUsed(cl));
                shuf = randperm(numel(trInd));
                trainCl = trInd(shuf(1:numTrToUse))';
                testCl = trInd(shuf(numTrToUse+1:numTrToUse*2))';
                
                trialsTrain = [trialsTrain trainCl];
                trialsTest = [trialsTest testCl];
                labelsTrain = [labelsTrain ones(1,numTrToUse)*classesUsed(cl)];
                labelsTest = [labelsTest ones(1,numTrToUse)*classesUsed(cl)];
            end
            % pull the corresponding trials
            x_train = x(trialsTrain,:);
            x_test = x(trialsTest,:);
            
            % for each problem, pull relevant classes, train and test
            for pb = 1:numProb
                pbclasses = probClasses(pb,:);
                
                % pull the trials for this problem
                train_x_pb = x_train(ismember(labelsTrain, pbclasses),:);
                test_x_pb = x_test(ismember(labelsTest, pbclasses),:);
                % pull the labels for this problem
                train_labels_pb = labelsTrain(ismember(labelsTrain, pbclasses));
                test_labels_pb = labelsTest(ismember(labelsTest, pbclasses));
                
                %dimensionality reduction
                if useSVD == 1
                    %dimensionality reduction: only take top n SVD features
                    nonNanInds = find(~isnan(train_x_pb(:,1))); % find the trials with all NaNs
                    % mean-center the non-nan data
                    xtrain_c = train_x_pb(nonNanInds,:) - mean(train_x_pb(nonNanInds,:));
                    xtest_c = test_x_pb(nonNanInds,:) - mean(train_x_pb(nonNanInds,:));
                    [Y,Sig,X] = svd(xtrain_c'); %numspks x time
                    sig = diag(Sig); %to view overall dim importance
                    train_x_pb = nan(size(train_x_pb,1),numSVs);
                    test_x_pb = nan(size(test_x_pb,1),numSVs);
                    %project spike data onto first N dimensions
                    train_x_pb(nonNanInds,:) = xtrain_c*Y(:,1:numSVs);
                    %project onto test data as well
                    test_x_pb(nonNanInds,:) = xtest_c*Y(:,1:numSVs);
                end
                
                mdl = fitcdiscr(train_x_pb,train_labels_pb);
                yhat = predict(mdl,test_x_pb);
                acc_half(pb,rep,bn) = length(find(test_labels_pb'-yhat==0))/length(test_labels_pb);
                
                %50/50 cross validation - null distribution
                yshuff_train = train_labels_pb(randperm(length(train_labels_pb)))';
                yshuff_test = test_labels_pb(randperm(length(test_labels_pb)))';
                
                mdl = fitcdiscr(train_x_pb,yshuff_train);
                yhat = predict(mdl,test_x_pb);
                nullacc_half(pb,rep,bn) = length(find(yshuff_test-yhat==0))/length(yshuff_test);
                
            end
        end
    end
end

% plot all accuracies in a heatmap for every bin 
% and add asterisks if it is outside of the bounds of the null accuracies

f2 = plotConfMats(acc_half,nullacc_half, conds_mod, sgt, numBins, binWidth, 8:12, probClasses_g);

% get numbers for paper
% accs and CIs in 0.5
disp(['0-0.5t: *FPa v *BLa: ' num2str(mean(acc_half(2,:,9))) ' ['...
    num2str(prctile(acc_half(2,:,9),2.5)) '-'...
    num2str(prctile(acc_half(2,:,9),97.5)) ']'])

disp(['0-0.5t: *FPf v *BLf: ' num2str(mean(acc_half(10,:,9))) ' ['...
    num2str(prctile(acc_half(10,:,9),2.5)) '-'...
    num2str(prctile(acc_half(10,:,9),97.5)) ']'])

disp(['0-0.5t: *BLa v *BLf: ' num2str(mean(acc_half(16,:,9))) ' ['...
    num2str(prctile(acc_half(16,:,9),2.5)) '-'...
    num2str(prctile(acc_half(16,:,9),97.5)) ']'])

disp(['0.5-1t: *BLa v *BLf: ' num2str(mean(acc_half(16,:,10))) ' ['...
    num2str(prctile(acc_half(16,:,10),2.5)) '-'...
    num2str(prctile(acc_half(16,:,10),97.5)) ']'])
% 
% t test comparing all accuracies
rdm1 = squeeze(mean(acc_half(:,:,9),2)); %0-0.5 time bin
rdm2 = squeeze(mean(acc_half(:,:,10),2)); %0.5-1 time bin
[~, p] = ttest(rdm1, rdm2, 'Tail','right');
disp(['one-tailed pairwise t test comparing 0-0.5 time bin and 0.5-1 time bin: '...
    num2str(p)])

%% plot accuracy matrix
% average the runs together to construct a symmetric accuracy matrix, then compare
% to the accuracy matrix for the null distribution
% assign asterisks to every point where the accuracy is significantly
% different from the null distribution
function fig = plotConfMats(acc_half_confMat, null_acc_confMat, conds, sgt, numBins, binWidth, binsToPlot, probClasses)

mean_accs= squeeze(mean(acc_half_confMat, 2)); % %mean across reps, becomes bin x actual x guess
postStimBinLen = 40;
preStimBinLen = 80;
%get the upper and lower CIs of the acc distribution for each grid point
meanAccMap = ones(numBins,max(max(probClasses)),max(max(probClasses)));
for bn = binsToPlot
        for pb = 1:size(probClasses,1)
            cr = probClasses(pb,1);
            cc = probClasses(pb,2);
            
            % transform into heatmap
            meanAccMap(bn,cr,cc) = mean_accs(pb,bn);
            meanAccMap(bn,cc,cr) = mean_accs(pb,bn); % it's symmetric
            
            acc_CI05(bn,cr,cc,:) = [prctile(acc_half_confMat(pb,:,bn),2.5) prctile(acc_half_confMat(pb,:,bn),97.5)];
            acc_CI01(bn,cr,cc,:) = [prctile(acc_half_confMat(pb,:,bn),.5) prctile(acc_half_confMat(pb,:,bn),99.5)];
            acc_CI001(bn,cr,cc,:) = [prctile(acc_half_confMat(pb,:,bn),.05) prctile(acc_half_confMat(pb,:,bn),99.95)];
            nullacc_CI05(bn,cr,cc,:) = [prctile(null_acc_confMat(pb,:,bn),2.5) prctile(null_acc_confMat(pb,:,bn),97.5)];
            nullacc_CI01(bn,cr,cc,:) = [prctile(null_acc_confMat(pb,:,bn),.5) prctile(null_acc_confMat(pb,:,bn),99.5)];
            nullacc_CI001(bn,cr,cc,:) = [prctile(null_acc_confMat(pb,:,bn),.05) prctile(null_acc_confMat(pb,:,bn),99.95)];
            % determine which grids deserve asterisks
            % check if null is smaller than acc
            if nullacc_CI05(bn, cr,cc,2)<=acc_CI05(bn, cr,cc, 1) % is real acc sig. greater than null acc?
                if nullacc_CI01(bn, cr,cc,2)<=acc_CI01(bn, cr,cc, 1)
                    if  nullacc_CI001(bn, cr,cc,2)<=acc_CI001(bn, cr,cc, 1)
                        CIsig_eff{bn,cr,cc} = '***'; CIsig_eff{bn,cc,cr} = '***';
                    else
                        CIsig_eff{bn,cr,cc} = '**'; CIsig_eff{bn,cc,cr} = '**';
                    end
                else
                    CIsig_eff{bn,cr,cc} = '*'; CIsig_eff{bn,cc,cr} = '*';
                end
            else
                CIsig_eff{bn,cr,cc} = ''; CIsig_eff{bn,cc,cr} = '';
            end
            
            % check if null is greater than acc
            if nullacc_CI05(bn, cr,cc,1)>=acc_CI05(bn, cr,cc, 2)
                if nullacc_CI01(bn, cr,cc,1)>=acc_CI01(bn, cr,cc, 2)
                    if  nullacc_CI001(bn, cr,cc,1)>=acc_CI001(bn, cr,cc, 2)
                        CIsig_eff{bn,cr,cc} = '***'; CIsig_eff{bn,cc,cr} = '***';
                    else
                        CIsig_eff{bn,cr,cc} = '**'; CIsig_eff{bn,cc,cr} = '**';
                    end
                else
                    CIsig_eff{bn,cr,cc} = '*'; CIsig_eff{bn,cc,cr} = '*';
                end
            end
        end
end
chance = 0.5;
numBinsPreTouch = numBins*(preStimBinLen/(preStimBinLen+postStimBinLen));
preTouchLb = (1:numBinsPreTouch)*binWidth;
preTouchLb = -preTouchLb(end:-1:1);
postTouchLb = (1:numBins - numel(preTouchLb))*binWidth;
Lbs = [preTouchLb 0 postTouchLb];


numCond = numel(conds);
fig = figure('Position',[50 50 1800 550]);
cmap = [linspace(1,0,100)' linspace(1,50/255,100)' linspace(1,255/255,100)'];
totBins = numel(binsToPlot);
for bn = binsToPlot
    cf = squeeze(meanAccMap(bn,:,:));
    subplot(1,totBins,bn-binsToPlot(1)+1)
    imagesc(cf)
    yticks(1:1:numCond)
    xticks(1:1:numCond)
    xticklabels(conds)
    yticklabels(conds)
    ax = gca(fig);
    ax.XAxis.FontSize = 7; ax.YAxis.FontSize = 7;
    xtickangle(-90)
    title(['S1: ' num2str(Lbs(bn)) ' - ' num2str(Lbs(bn+1)) 's'])
    caxis([chance 1])
    colormap(cmap);
    cb = colorbar('northoutside');
    cb.Ticks = [0.5 0.75 1] ; %Create 8 ticks from zero to 1
    axis square
    axis tight
    for cr = 1:numCond
        for cc = 1:numCond
            if size(CIsig_eff{bn,cr,cc},2)>0 % if significant
                if size(CIsig_eff{bn,cr,cc},2)<3 % just to make it pretty, format *** differently
                    text(cc,cr+0.1,sprintf('%s',CIsig_eff{bn,cr,cc}), 'FontSize', 10, 'HorizontalAlignment', 'center');
                else
                    text(cc,cr,sprintf('%s','**'), 'FontSize', 10, 'HorizontalAlignment', 'center');
                    text(cc,cr+0.3,sprintf('%s','*'), 'FontSize', 10, 'HorizontalAlignment', 'center');
                end
            end
        end
    end
end
sgtitle([sgt ' accuracy heatmaps'])
end