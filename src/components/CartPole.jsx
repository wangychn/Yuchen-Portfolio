import React, { useEffect, useRef, useState } from 'react';
import * as ort from 'onnxruntime-web';
ort.env.wasm.wasmPaths = {
    'ort-wasm.wasm': '/cartpole/ort-wasm.wasm',
    'ort-wasm-simd.wasm': '/cartpole/ort-wasm-simd.wasm',
    'ort-wasm-threaded.wasm': '/cartpole/ort-wasm-threaded.wasm',
    'ort-wasm-simd-threaded.wasm': '/cartpole/ort-wasm-simd-threaded.wasm',
    'ort-wasm.wasm.mem': '/cartpole/ort-wasm.wasm.mem',
};

ort.env.wasm.numThreads = 1;

import { CartPoleEnv, CartPoleRenderer } from '../lib/cartpole.js';

export default function CartPole({ modelPath = '/cartpole/model.onnx' }) {
    const canvasRef = useRef(null);

    const [actionState, setActionState] = useState(0); // 0 = left, 1 = right
    const [flashKey, setFlashKey] = useState(0);       // to retrigger CSS animation
    const scoresRef = useRef([0, 0]);
    const lastActionRef = useRef(0);

    // percentages for bar widths (computed each render from ref)
    const leftPct = Math.max(0, Math.min(100, (scoresRef.current?.[0] ?? 0) * 100));
    const rightPct = Math.max(0, Math.min(100, (scoresRef.current?.[1] ?? 0) * 100));


    useEffect(() => {
        let isMounted = true;
        let session;
        let inputName;
        let outputName;
        const run = async () => {
            const canvas = canvasRef.current;
            if (!canvas) return;

            const env = new CartPoleEnv();
            const renderer = new CartPoleRenderer(canvas, env);
            // Draw the initial state
            renderer.draw(env.reset());

            // Load the model once
            session = await ort.InferenceSession.create(modelPath);
            // Use the model's actual input/output names
            inputName = session.inputNames[0];
            outputName = session.outputNames[0];

            let input = new ort.Tensor(
                'float32',
                Float32Array.from(env.state),
                [1, env.state.length]
            );

            // Main loop (~20 FPS)
            while (isMounted) {
                // Run inference with the correct key
                const outputMap = await session.run({ [inputName]: input });

                // Read the output tensor by its actual name
                const tensor = outputMap[outputName];
                const scores = Array.from(tensor.data);
                const action = scores.indexOf(Math.max(...scores));

                // Update HUD data with cooldown
                const now = Date.now();
                const cooldownMs = 200; // half a second
                if (now - lastActionRef.current > cooldownMs) {
                    scoresRef.current = scores;
                    setActionState(action);
                    setFlashKey(k => k + 1);
                    lastActionRef.current = now;
                }
                // (removed unconditional update of scoresRef.current, setActionState, setFlashKey)

                const { state: nextState, reward, done } = env.step(action);
                renderer.draw(nextState);
                input = new ort.Tensor(
                    'float32',
                    Float32Array.from(nextState),
                    [1, nextState.length]
                );

                if (done) env.reset();

                await new Promise(r => setTimeout(r, 25));
            }
        };

        run();

        return () => {
            isMounted = false;
        };
    }, [modelPath]);

    return (
        <div style={{ display: "flex", flexDirection: "row", width: 600, height: 400 }}>
            <canvas
                ref={canvasRef}
                width={1200}
                height={800}
                style={{ display: "block" }}
            />

            {/* HUD overlay in the bottom-right corner */}
            <div
                className={`cp-hud ${actionState === 1 ? "right" : "left"}`}
                key={flashKey}
                style={{
                    position: "absolute",
                    bottom: 100,
                    right: 16,
                    borderRadius: 8,
                    fontFamily: "var(--font-body, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif)",
                    fontSize: 14,
                    background: "rgba(31, 21, 21, 0.65)",
                    color: "#fff",
                    userSelect: "none",
                    pointerEvents: "none",
                    minWidth: 200,
                    boxShadow: "0 2px 8px rgba(0,0,0,0.2)",
                    animation: "cpFlash 240ms ease-in-and-out",
                    padding: 10
                }}
            >
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
                    <strong>Model Output</strong>
                    <span style={{
                        padding: "2px 6px",
                        borderRadius: 6,
                        background: actionState === 1 ? "rgba(76,175,80,0.9)" : "rgba(33,150,243,0.9)"
                    }}>
                        {actionState === 1 ? "RIGHT" : "LEFT"}
                    </span>
                </div>

                {/* Bars for scores/Q-values */}
                <div style={{ marginBottom: 4 }}>
                    <div style={{ display: "flex", justifyContent: "space-between" }}>
                        <span>Left Q Val</span>
                        <span>{scoresRef.current?.[0]?.toFixed?.(2) ?? "-"}</span>
                    </div>
                    <div style={{ height: 6, background: "rgba(255,255,255,0.15)", borderRadius: 4, overflow: "hidden" }}>
                        <div style={{
                            height: "100%",
                            width: `${leftPct}%`,
                            background: "#2196f3",
                            transition: "width 120ms linear"
                        }} />
                    </div>
                </div>

                <div>
                    <div style={{ display: "flex", justifyContent: "space-between" }}>
                        <span>Right Q Val</span>
                        <span>{scoresRef.current?.[1]?.toFixed?.(2) ?? "-"}</span>
                    </div>
                    <div style={{ height: 6, background: "rgba(255,255,255,0.15)", borderRadius: 4, overflow: "hidden" }}>
                        <div style={{
                            height: "100%",
                            width: `${rightPct}%`,
                            background: "#4caf50",
                            transition: "width 120ms linear"
                        }} />
                    </div>
                </div>
            </div>
        </div>
    );
}