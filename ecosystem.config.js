module.exports = {
  apps : [{
    name: "AI-Web",
    script: "../app.py",
    interpreter: "/opt/miniconda3/envs/dapeng/bin/python3",
    log: "../logs/web.log",
    time: true
  },{
    name: "AI-Model",
    script: "multi_tracker.py",
    interpreter: "/opt/miniconda3/envs/dapeng/bin/python3",
    log: "../logs/ai_model.log",
    time: true,
    // output: "./output.avi"
  }]
};
