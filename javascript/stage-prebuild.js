/**
 *  @file javascript/stage-prebuild.js
 *  @author Ash Vardanian
 *  @date September 9, 2026
 *  @brief Copies the addon `cmake-js` produced into the layout `node-gyp-build` resolves at install time:
 *      `prebuilds/<platform>-<arch>/node.napi.node`, whose name is how it picks the matching ABI.
 */
const fs = require("fs");
const path = require("path");

const [, , buildDirectory = "build_node", architecture = process.arch] = process.argv;
const source = path.join(buildDirectory, "Release", "usearch.node");
const target = path.join("prebuilds", `${process.platform}-${architecture}`);

fs.mkdirSync(target, { recursive: true });
fs.copyFileSync(source, path.join(target, "node.napi.node"));
console.log(`staged ${source} -> ${path.join(target, "node.napi.node")}`);
