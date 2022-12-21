% figure2bc_plotRDMandMDS
% plots RDM and MDS plots in figure2b and 2c respectively
% Isabelle Rosenthal 2022

filePath = '..\Data\';
chList = {1:96};

% load in preprocessed data file
allData = load(['..\Data\touchExploration_preprocessedSpks.mat']);
trialClass = allData.trialClass;
trialEffector = allData.trialEffector;
trialType = allData.trialType;
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

effLbs = unique(label_eff);
modLbs = unique(label_mod);
tyLbs = unique(label_ty);
mod2useInds = 1:numel(label_eff);

% load in the python saved RSA file, which can be generated with 
% figure2b_constructRDMs.py
pyRDM = load(['..\Data\touchExploration_RSA.mat']);

condRDM = pyRDM.RDMs; % bin x conditions

% set up timings
postStimBinLen = 40;
preStimBinLen = 80;
binWidth = 0.5; % in sec
binStarts = [1:10:111]; % every 0.5 sec
binEnds = [10:10:120]; % every 0.5 sec
win = 0.05;
numBins = numel(binStarts);
numBinsPreTouch = numBins*(preStimBinLen/(preStimBinLen+postStimBinLen));

preTouchLb = (1:numBinsPreTouch)*binWidth;
preTouchLb = -preTouchLb(end:-1:1);
postTouchLb = (1:numBins - numel(preTouchLb))*binWidth;
Lbs = [preTouchLb 0 postTouchLb];

% order the conditions appropriately
modLbs = [1 4 10 13 8 11 2 5 7];
conds_mod = {'FPa',  'FPf', 'BLa','BLf',...
    'VrFPa','VrFPf','TPa','TPf','obj'};
tyLbs = [1 7 5 2 4]; 
conds_ty = {'FP','BL','VrFP', 'TP', 'obj'};
effLbs = [1:3];
conds_eff = {'arm','finger','obj'};
Mod2Ty = [1 1 7 7 5 5 2 2 4];
Mod2Eff = [1 2 1 2 1 2 1 2 3];
tySort = [1, 3, 7, 9, 6, 8, 2, 4, 5]; % to deal with rank order RDM
[~,rankSort] = sort(modLbs);
mod2TyRank = Mod2Ty(rankSort);
mod2EffRank = Mod2Eff(rankSort);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% run statistical tests on computed RDMs 

% load pairwise modality decoding to compare to RDMs
decodeData = load('../Data/40dimSVD_LDAdecoding_PairwiseMod_0.5sbins.mat');
pairwiseAccs = decodeData.acc_half;
classesOrd = decodeData.probClasses_g;
clear decodeData

% % run pearson correlation to see if RDM at 0, 0.5, 1 are similar to their
% % respective pairwise decoding plots
% for bn = 1:3
%     decVec = squeeze(mean(pairwiseAccs(:,:,bn+8),2)); %0-0.5 time bin
%     rdmMat = squeeze(condRDM(bn+8,:,:)); rdmVec = []; %rdm is ordered numerically
%     for pa = 1:size(classesOrd,1)
%         cc = tySort(classesOrd(pa,1));
%         cr = tySort(classesOrd(pa,2));
%         rdmVec(pa) = rdmMat(cc,cr);
%     end
%     [r(bn) p(bn)] = corr(decVec, rdmVec');
% end
% p = bonf_holm(p);
% disp(['pearson corr (bf corrected) between Mod LDA and RDM at t[0 0.5]: r='...
%     num2str(r(1)) ', p=' num2str(p(1))])
% disp(['pearson corr (bf corrected) between Mod LDA and RDM at t[0.5 1]: r='...
%     num2str(r(2)) ', p=' num2str(p(2))])
% disp(['pearson corr (bf corrected) between Mod LDA and RDM at t[1 1.5]: r='...
%     num2str(r(3)) ', p=' num2str(p(3))])

% run pearson correlation between -0.5-0s bin and subsequent bins
rdmMatBefore = squeeze(condRDM(8,:,:));
rdmMat1 = squeeze(condRDM(9,:,:)); rdmMat2 = squeeze(condRDM(10,:,:));
rdmMat3 = squeeze(condRDM(11,:,:)); rdmMat4 = squeeze(condRDM(12,:,:));
[rc pc] = corrcoef(rdmMatBefore, rdmMat1);
r1(1) = rc(2,1); p1(1) = pc(2,1);
[rc pc] = corrcoef(rdmMatBefore, rdmMat2);
r1(2) = rc(2,1); p1(2) = pc(2,1);
[rc pc] = corrcoef(rdmMatBefore, rdmMat3);
r1(3) = rc(2,1); p1(3) = pc(2,1);
[rc pc] = corrcoef(rdmMatBefore, rdmMat4);
r1(4) = rc(2,1); p1(4) = pc(2,1);
p1 = bonf_holm(p1);
disp(['pearson corrs (bf corrected) between RDM -0.5 and subsequent RDMs:'])
disp(['rs=' num2str(r1)])
disp(['ps=' num2str(p1)])

% run pearson correlation between 0.5-0s bin and subsequent bins
[rc pc] = corrcoef(rdmMat1, rdmMat2);
r2(1) = rc(2,1); p2(1) = pc(2,1);
[rc pc] = corrcoef(rdmMat1, rdmMat3);
r2(2) = rc(2,1); p2(2) = pc(2,1);
[rc pc] = corrcoef(rdmMat1, rdmMat4);
r2(3) = rc(2,1); p2(3) = pc(2,1);
p2 = bonf_holm(p2);
disp(['pearson corrs (bf corrected) between RDM 0 and subsequent RDMs:'])
disp(['rs=' num2str(r2)])
disp(['ps=' num2str(p2)])

% within 0-0.5s bin, unpaired t-test on vis stim distances vs phys distances
% so all the distances within TP, obj, vFP on one side
% all the distances within rFP, BL on the other side
% order in rdm is conds_mod(rankSort)
distVecVis = [rdmMat1(2,4) rdmMat1(2,5) rdmMat1(2,6) rdmMat1(2,8)...
    rdmMat1(4,5) rdmMat1(4,6) rdmMat1(4,8)...
    rdmMat1(5,6) rdmMat1(5,8)...
    rdmMat1(6,8)];
distVecPhys = [rdmMat1(1,3) rdmMat1(1,7) rdmMat1(1,9) ...
    rdmMat1(3,7) rdmMat1(3,9) ...
    rdmMat1(7,9) ];
[~,p3] = ttest2(distVecVis, distVecPhys);
disp(['unpaired ttest between visual stim distances and phys stim distances, ps=' num2str(p3)])

% paired t test each touch type: rFP, BL. Within each, combine both locations
% and compare all distances to non-physical touch types vFP, TP, obj
rFPDists = [rdmMat1(1,[2 4 5 6 8])...
    rdmMat1(3,[2 4 5 6 8])];
vBLDists = [rdmMat1(6,[2 4 5 6 8])...
    rdmMat1(9,[2 4 5 6 8])];
[~,p4] = ttest(rFPDists,vBLDists);
disp(['paired ttest at 0 between rFP and vBL dists:, p=' num2str(p4)])
disp(['rFP dist to vis stim: avg=' num2str(mean(rFPDists)) ' std=' num2str(std(rFPDists))])
disp(['vBL dist to vis stim: avg=' num2str(mean(vBLDists)) ' std=' num2str(std(vBLDists))])

% % test clustering between arm and finger
% % t-test between FPa/FPf distance vs FPa and FPf distances to
% % within-location physical stim, and the same test on rBL and vBL
% acrossLocDist = rdmMat1(1,3); %rFP
% withinLocDists = [rdmMat1(1,[7 10]) rdmMat1(3,[9 11])];
% [~, p5(1)] = ttest(withinLocDists,acrossLocDist, 'tail','left');
% acrossLocDist = rdmMat1(10,11); %rBL
% withinLocDists = [rdmMat1(10,[1 7]) rdmMat1(11,[3 9])];
% [~, p5(2)] = ttest(withinLocDists,acrossLocDist, 'tail','left');
% acrossLocDist = rdmMat1(7,9); %vBL
% withinLocDists = [rdmMat1(9,[1 10]) rdmMat1(9,[3 11])];
% [~, p5(3)] = ttest(withinLocDists,acrossLocDist, 'tail','left');
% p5 = bonf_holm(p5);
% disp(['one-tailed ttest (bf corr) at 0 between rFP a/f dist and other phys dists:, p=' num2str(p5(1))])
% disp(['one-tailed ttest (bf corr) at 0 between rBL a/f dist and other phys dists:, p=' num2str(p5(2))])
% disp(['one-tailed ttest (bf corr) at 0 between vBL a/f dist and other phys dists:, p=' num2str(p5(3))])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% visualize the RDM for each nsp at each time bin - effector first
conds_names = conds_mod;
numConds = numel(modLbs);
resNumBins = 5; % only want to plot the bins around the ITI


% visualize the RDM at each time bin, sorting into proper order

% order of RDM from Python: 
% modLabels = [1, 2, 4, 5, 7, 8, 10, 11, 13, 15, 16]

f2 = figure('Position',[50 50 1800 550]);
sgt = 'touch-onset-aligned RDM';
cmap = hot(100);
for bn = 1:resNumBins
    bnNum = bn + 7;
    
    cf = squeeze(condRDM(bnNum,:,:));
    % rearrange RDM
    for cc = 1:size(cf,1)
        for cr = 1:size(cf,2)
            cf_ty(cc, cr) = cf(tySort(cc), tySort(cr));
        end
    end
    % rework so only lower diagonal is visible
    for cc = 1:size(cf_ty,1)
        for cr = 1:size(cf_ty,2)
            if cc<=cr
                cf_ty(cc,cr)=1; % so it will appear white
            end
        end
    end
    subplot(1,resNumBins,bn);
    imagesc(cf_ty);
    yticks(1:1:numConds);
    xticks(1:1:numConds);
    xticklabels(conds_mod);
    yticklabels(conds_mod);
    xtickangle(-90);
    title(['S1: ' num2str(Lbs(bnNum)) ' - ' num2str(Lbs(bnNum+1)) 's']);
    set(gca,'ColorScale','log')
    caxis([0.01 0.3])
    colormap(cmap);
    cc = colorbar('northoutside');
    axis square
    axis tight
end
sgtitle([sgt ' - Touch Type sorted, CV noise-corr Mahalanobis distance']);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% use the RDMs to build MDS plots
modLbs_rankOrd = unique([1 4 10 13 8 11 2 5 7]);
% using toolbox from https://github.com/rsagroup/rsatoolbox_matlab
userOptions = rsa.rsa_defineUserOptions_iar('TouchExploration_real');

localOptions.fontSize = 10; % for the labels on the MDS plot
userOptions.dotSize=20;
rdmStruct.name = 'test';
rdmStruct.color = [ 0 0 0];
resNumBins = 5; % only plot certain bins
effMarker = {'o','^','s'};
tyHues = [84/255 1/255 41/255;
    216/255 122/255 182/255;
    0/255 48/255 146/255;%  
    139/255 37/255 154/255;
    0.6 0.6 0.6;];
userOptions.conditionColours =[];userOptions.markerStyles =[];

for mod = 1:numel(tySort)
    ty = find(mod2TyRank(mod)==tyLbs); eff = find(mod2EffRank(mod)==effLbs);
    userOptions.conditionColours(mod,:)=tyHues(ty,:);
    userOptions.markerStyles{mod} = effMarker{eff};
end
rotateAngles = [95 90 85 93 70]*pi/180; % angle to rotate each bin, customized so that they all have similar orientation.

for bn = 1:resNumBins
    figure; fnum = get(gcf,'Number');
    localOptions.figureNumber = fnum;
    bnNum = bn + 7;
    for nsp = 3
        localOptions.rotateAngle = rotateAngles(bn); % rotation angle
        cf = squeeze(condRDM(bnNum,:,:));
        numNeg = (sum(sum(cf<0))-16)/2; % divide because symmetric and remove diagonal
        cf(cf<=0) = 0.0000000001; % now take out the zeros
        % normalize so in the 0-1 range
        cf = cf/max(max(cf));
        rdmStruct.RDM = cf;
        [pearsonR(bn), p_pearson(bn), pointCoords] = rsa.MDSConditions_iar(rdmStruct, userOptions, localOptions);
        set(gcf, 'Position',[508   462   608   484]);
    end
end
% print pearsonR and p for paper
disp('Pearson Rs for MDS plots:')
disp(pearsonR)
disp('Pearson ps for MDS plots:')
disp(p_pearson)