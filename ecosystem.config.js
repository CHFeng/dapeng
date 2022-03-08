module.exports = {
  apps : [{
    name: "AI-Web",
    script: "ai_web",
    // interpreter: "/opt/miniconda3/envs/dapeng/bin/python3",
    log: "logs/web.log",
    time: true
  },{
    name: "AI-Model",
    script: "./ai_model/multi_tracker",
    // interpreter: "/opt/miniconda3/envs/dapeng/bin/python3",
    log: "logs/ai_model.log",
    time: true,
    args: "--dont_show=true"
  }]
};
