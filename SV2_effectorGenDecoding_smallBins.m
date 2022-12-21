% figure3a_effectorGenDecoding
% LDA decoding on top PCs of multi unit firing rates,
% generalizing touch effector (arm or finger) decoding across touch types,
% uses 0.1s bins instead of 0.5s bins,
% then plots the decoding results as a movie which is saved out, see
% Supplemental Video 3
% Isabelle Rosenthal 2022

% parameters
videoSavePath = pwd; % set this to where you want the video to save
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
chList = {1:96};

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

% order the conditions the way you want (no object condition)
modLbs = [1 4 10 13 8 11 2 5];
conds_mod = {'FPa',  'FPf', 'BLa','BLf',...
    'VrFPa','VrFPf','TPa','TPf'};
tyLbs = [1 7 5 2]; %
conds_ty = {'FP','BL','VrFP', 'TP'};
effLbs = [1:2];
conds_eff = {'arm','finger'};
Mod2Ty = [1 1 7 7 5 5 2 2];
Mod2Eff = [1 2 1 2 1 2 1 2];
mod_numMap = [1:numel(modLbs)];

% pick times and bins to use
postStimBinLen = 40;
preStimBinLen = 80;
binWidth = 0.1; % for decoding in sec, every 100ms
binStarts = [1:2:59]; % every 0.1 sec
binEnds = [2:2:60]; % every 0.1 sec
numBins = numel(binStarts);

% set name of analysis
sgt = 'touch-aligned LDA, effector gen';
if useSVD == 1
    sgt = [sgt ' on top ' num2str(numSVs) ' dims (SVD)'];
else
    sgt = [sgt ' on all channels'];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if toLoad == 1
    % if you're loading in former results
    load(['..\Data\40dimSVD_LDAdecoding_effGen_0.1sbins.mat']);
    disp('results loaded successfully')
else
    % effector generalizing over touch type decoding
    clear binnedSpks nullacc_half  acc_half
    
    % define individual pairwise problems
    armClasses = [1 2 3 8 10]; fingerClasses = [4 5 6 11 13];
    probs_byTy_g = nchoosek(1:numel(conds_ty),2);
    probs_byTy_g = [probs_byTy_g; probs_byTy_g(:,2) probs_byTy_g(:,1)]; % add opposite train/test pattern
    probs_byTy = probs_byTy_g;
    % remap
    for mod = 1:numel(tyLbs)
        probs_byTy(probs_byTy_g(:,1)==mod,1) = tyLbs(mod);
        probs_byTy(probs_byTy_g(:,2)==mod,2) = tyLbs(mod);
    end
    numProb = size(probs_byTy,1);
    for pI = 1:numProb
        probNames{pI} = ['AvsFtrain' conds_ty{probs_byTy_g(pI,1)} 'test' conds_ty{probs_byTy_g(pI,2)}];
        probTrainClasses_g(pI,:) = find(Mod2Ty==probs_byTy(pI,1));
        probTestClasses_g(pI,:) = find(Mod2Ty==probs_byTy(pI,2));
        % remap where needed
        probTrainClasses(pI,:) = modLbs(find(Mod2Ty==probs_byTy(pI,1)));
        probTestClasses(pI,:) = modLbs(find(Mod2Ty==probs_byTy(pI,2)));
    end


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
    classesUsed = unique([probTrainClasses; probTestClasses]);
    for cl = 1:numel(classesUsed)
        trCount(cl) = sum(label_mod==classesUsed(cl));
    end
    numTrToUse = round(min(trCount)/2); %per train or test
    for bn = 1:numel(binStarts)
        disp(['running bin ' num2str(bn) '/' num2str(numel(binStarts))]);
        % sort binnedSpks into appropriate bins and mean
        bnspk = nanmean(binnedSpks_noC(:,:,binStarts(bn):binEnds(bn)),3);
        
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
                trainClasses = probTrainClasses(pb,:);
                testClasses = probTestClasses(pb,:);
                
                % pull the trials for this problem
                train_x_pb = x_train(ismember(labelsTrain, trainClasses),:);
                test_x_pb = x_test(ismember(labelsTest, testClasses),:);
                % pull the labels for this problem
                train_labels_pb = labelsTrain(ismember(labelsTrain, trainClasses));
                test_labels_pb = labelsTest(ismember(labelsTest, testClasses));
                %remap the labels to be matching
                % all arm labels = 1
                % all finger labels = 2
                train_labels_pb(ismember(train_labels_pb, armClasses)) = 1;
                test_labels_pb(ismember(test_labels_pb, armClasses)) = 1;
                train_labels_pb(ismember(train_labels_pb, fingerClasses)) = 2;
                test_labels_pb(ismember(test_labels_pb, fingerClasses)) = 2;
                
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
% make a movie of this over time

numProb = numel(probNames);
trainType = probs_byTy_g(:,1);
testType = probs_byTy_g(:,2);
f2 = plotConfMats(acc_half,nullacc_half, conds_ty, sgt, numBins, binWidth,...
    1:30, trainType, testType);

% create the video writer with 1 fps
writerObj = VideoWriter([ videoSavePath '\SV3_PairwiseModalityDecoding.avi']);
writerObj.FrameRate = 1;
% open the video writer
open(writerObj);
% write the frames to the video
for i=1:length(f2)
    % convert the image to a frame
    frame = f2(i) ;
    writeVideo(writerObj, frame);
end
% close the writer object
close(writerObj);
%% plot confusion matrix
% average the runs together to construct a confusion matrix, then compare
% to the confusion matrix for the null distribution
% assign asterisks to every point where the accuracy is significantly
% different from the null distribution
function F = plotConfMats(acc_half_confMat, null_acc_confMat, conds, sgt, numBins, binWidth, binsToPlot, trainType, testType)

mean_accs= squeeze(mean(acc_half_confMat, 2)); % %mean across reps, becomes bin x nsp x actual x guess
postStimBinLen = 40;
preStimBinLen = 20;
%get the upper and lower CIs of the acc distribution for each grid point
meanAccMap = ones(numBins,numel(conds),numel(conds));
for bn = binsToPlot
    for pb = 1:size(trainType,1)
        cr = trainType(pb);
        cc = testType(pb);
        
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
        if nullacc_CI05(bn, cr,cc,2)<acc_CI05(bn, cr,cc, 1) % is real acc sig. greater than null acc?
            if nullacc_CI01(bn, cr,cc,2)<acc_CI01(bn, cr,cc, 1)
                if  nullacc_CI001(bn, cr,cc,2)<acc_CI001(bn, cr,cc, 1)
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
        if nullacc_CI05(bn, cr,cc,1)>acc_CI05(bn, cr,cc, 2)
            if nullacc_CI01(bn, cr,cc,1)>acc_CI01(bn, cr,cc, 2)
                if  nullacc_CI001(bn, cr,cc,1)>acc_CI001(bn, cr,cc, 2)
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
cmap = [linspace(1,0,100)' linspace(1,50/255,100)' linspace(1,255/255,100)'];
for bn = binsToPlot
    fig = figure('Position',[50 50 700 550]);
    cf = squeeze(meanAccMap(bn,:,:));
    imagesc(cf)
    yticks(1:1:numCond)
    xticks(1:1:numCond)
    xticklabels(conds)
    yticklabels(conds)
    ylabel('train')
    xlabel('test')
    ax = gca(fig);
    ax.XAxis.FontSize = 7; ax.YAxis.FontSize = 7;
    xtickangle(-90)
    caxis([chance 1])
    colormap(cmap);
    colorbar('northoutside')
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
    title(sprintf([sgt ' accuracies \n bin: ' num2str(Lbs(bn)) 's']))
    F(bn) = getframe(gcf) ;
    close(fig)
end
end

