load('..\columbia.mat');
SET_SIZE = 30000;
net = newff(alphabet, targets, [130 100]);
net = init(net);
net.layers{1}.transferFcn = 'logsig';
net.layers{2}.transferFcn = 'logsig';
net.layers{3}.transferFcn = 'logsig';
net.inputs{1}.processFcns = {};
net.outputs{3}.processFcns = {};
  
net.userdata.report_interval = 1;
net.userdata.num_threads = 10;
net.userdata.learning_rate = 0.7;
net.trainParam.goal = 1e-8;
net.trainParam.time = 600;
net.trainParam.epochs = 10;

[net, log] = traingd2(net, alphabet(:,1:SET_SIZE), targets(:,1:SET_SIZE));
