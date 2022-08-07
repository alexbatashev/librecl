module.exports = ({ core, process }) => {
  const fs = require('fs');
  fs.readFile('./test_configs.json', 'utf8', (err, data) => {
    if (err) {
      console.log(`Error reading file from disk: ${err}`);
    } else {
      const inputs = JSON.parse(process.env.GHA_INPUTS);
      const configs = JSON.parse(data);

      const requiredConfig = configs[inputs["test-config"]];

      let configStr = JSON.stringify(requiredConfig);
      configStr = configStr.replaceAll("${{ github.workspace }}", process.env.GITHUB_WORKSPACE);

      core.setOutput('matrix', configStr);
    }
  });
};
