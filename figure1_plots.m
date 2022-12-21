% figure1_plots
% plots firing rates averaged by task conditions in individual channels
% Isabelle Rosenthal 2022

ch = 39; % pick the channel to display (39 in the paper)

% load in preprocessed files
allData = load(['..\Data\touchExploration_preprocessedSpks.mat']);
dataN = allData.dataN;
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

% remove catch trials
label_eff = labelc_eff(labelc_eff~=0);
label_mod = labelc_mod(labelc_mod~=0);
label_ty = labelc_ty(labelc_ty~=0);

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Look at single channel activity

% 2 sec pre touch onset, 2 sec post touch onset
postStimBinLen = 40;
preStimBinLen = 40;
xTickLbs = -2:0.5:2; % for x ticks in seconds

% visualization parameters
smoothFactor = 3; % level of smoothing for plotting lines
alpha = 0.2; % transparency of standard error lines
effC = [0.1 0.1 0.1];
effStyles = {'-','--',':'};
tyHues = [84/255 1/255 41/255;
    216/255 122/255 182/255;
    0/255 48/255 146/255;
    139/255 37/255 154/255;
    0.6 0.6 0.6;];


%get spike data
spk = squeeze(dataN(:,ch,:,:));
%reshape across sessions
spk = reshape(spk,[],2);
%now trim all ITIs to be the same length
spk(:,1) = cellfun(@(x) (x(max(1,size(x,2)-preStimBinLen+1):end)),spk(:,1),'un',0);
% there's one trial that's too short so nan-pad it in the ITI
spk(:,1) = cellfun(@(x) ([nan(1, max(preStimBinLen - size(x,2),0)) x]), spk(:,1),'un',0);
spk = cell2mat(spk); % trial x time bin

% %smooth spike firing just overall
spk = sgolayfilt(spk,1,smoothFactor,[],2);

spk = spk(labelc_mod~=0,:); %remove catch trials

f1 = figure('Position', [500 400 800 300], 'Color',[1 1 1]);

%view single channel activity for each effector
for eff = 1:numel(effLbs)
    a = 1:size(spk,2);
    b = nanmean(spk(label_eff==effLbs(eff),:),1);
    c = 1.96*(nanstd(spk(label_eff==effLbs(eff),:),[],1)/sqrt(length(find(label_eff==effLbs(eff)))));
    
    keepIndex = ~isnan(b) & ~isnan(c);
    a = a(keepIndex);
    b = b(keepIndex);
    c = c(keepIndex);
    if effLbs(eff)==3 %obj
        [~,ef(eff)] = utils.plotsem_alpha(a,b,c,tyHues(5,:),tyHues(5,:),alpha, effStyles{eff});
    else
        [~,ef(eff)] = utils.plotsem_alpha(a,b,c,effC,effC,alpha, effStyles{eff});
    end
    hold on
    xlim([1 size(spk,2)])
    % ylim([min(b)-max(c) max(b)+max(c)])
    %set(gca,'YTickLabel',[]);
    %gcaExpandable
end
ylim([0 8.5])
line([preStimBinLen preStimBinLen],[ylim],'color',[0 0 0])
line([preStimBinLen-10 preStimBinLen-10],[ylim],'color',[0 0 0],'LineStyle','--')
line([preStimBinLen+20 preStimBinLen+20],[ylim],'color',[0 0 0])
legend(ef, conds_eff, 'Location',  'eastoutside')
title(['S1: ch ' num2str(ch) ': effector across touch types'])
xticks(0:10:preStimBinLen+postStimBinLen)
xticklabels(xTickLbs); % in sec

%view single channel activity for each type
cmap = tyHues;
f2 = figure('Position', [500 400 800 300], 'Color',[1 1 1]);
for ty = 1:numel(tyLbs)
    a = 1:size(spk,2);
    b = nanmean(spk(label_ty==tyLbs(ty),:),1);
    c = 1.96*(nanstd(spk(label_ty==tyLbs(ty),:),[],1)/sqrt(length(find(label_ty==tyLbs(ty)))));
    
    keepIndex = ~isnan(b) & ~isnan(c);
    a = a(keepIndex);
    b = b(keepIndex);
    c = c(keepIndex);
    
    if tyLbs(ty)==4 %obj
        [~, tpy(ty)] = utils.plotsem_alpha(a,b,c,cmap(ty,:),cmap(ty,:),alpha, effStyles{3});
    else
        [~, tpy(ty)] = utils.plotsem_alpha(a,b,c,cmap(ty,:),cmap(ty,:),alpha, '-');
    end
    hold on
    xlim([1 size(spk,2)])
    % ylim([min(b)-max(c) max(b)+max(c)])
    %set(gca,'YTickLabel',[]);
    %gcaExpandable
end
ylim([0 8.5])
line([preStimBinLen preStimBinLen],[ylim],'color',[0 0 0])
line([preStimBinLen-10 preStimBinLen-10],[ylim],'color',[0 0 0],'LineStyle','--')
line([preStimBinLen+20 preStimBinLen+20],[ylim],'color',[0 0 0])
legend(tpy, conds_ty, 'Location', 'eastoutside')
title(['S1: ch ' num2str(ch) ': touch type across effectors'])
xticks(0:10:preStimBinLen+postStimBinLen)
xticklabels(xTickLbs); % in sec

%view single channel activity for each modality
f3 = figure('Position', [500 400 800 600], 'Color',[1 1 1]);
for modality = 1:numel(modLbs)
    ty = find(Mod2Ty(modality)==tyLbs); eff = find(Mod2Eff(modality)==effLbs);
    a = 1:size(spk,2);
    b = nanmean(spk(label_mod==modLbs(modality),:),1);
    c = 1.96*(nanstd(spk(label_mod==modLbs(modality),:),[],1)/sqrt(length(find(label_mod==modLbs(modality)))));
    
    keepIndex = ~isnan(b) & ~isnan(c);
    a = a(keepIndex);
    b = b(keepIndex);
    c = c(keepIndex);
    
    [~, md(modality)] = utils.plotsem_alpha(a,b,c,tyHues(ty,:),tyHues(ty,:),alpha, effStyles{eff});
    hold on
    xlim([1 size(spk,2)])
    % ylim([min(b)-max(c) max(b)+max(c)])
    %set(gca,'YTickLabel',[]);
    %gcaExpandable
end
ylim([0 8.5])
line([preStimBinLen preStimBinLen],[ylim],'color',[0 0 0])
line([preStimBinLen-10 preStimBinLen-10],[ylim],'color',[0 0 0],'LineStyle','--')
line([preStimBinLen+20 preStimBinLen+20],[ylim],'color',[0 0 0])
%FP = first persomn TP = third person
legend(md,conds_mod, 'Location', 'eastoutside')
title(['S1: ch ' num2str(ch) ': individual modalities'])
xticks(0:10:preStimBinLen+postStimBinLen)
xticklabels(xTickLbs); % in sec

clear spk label