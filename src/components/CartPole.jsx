import React, { useEffect, useRef } from 'react';
import * as ort from 'onnxruntime-web';
ort.env.wasm.wasmPaths = {
    'ort-wasm.wasm': '/cartpole/ort-wasm.wasm',
    'ort-wasm-simd.wasm': '/cartpole/ort-wasm-simd.wasm',
    'ort-wasm-threaded.wasm': '/cartpole/ort-wasm-threaded.wasm',
    'ort-wasm-simd-threaded.wasm': '/cartpole/ort-wasm-simd-threaded.wasm',
    'ort-wasm.wasm.mem': '/cartpole/ort-wasm.wasm.mem',
};

ort.env.wasm.numThreads = 1;

// Your JS port of the CartPole environment and drawing logic
import { CartPoleEnv, CartPoleRenderer } from '../lib/cartpole.js';

export default function CartPole({ modelPath = '/cartpole/model.onnx' }) {
    const canvasRef = useRef(null);

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
                console.log('model output:', tensor.data, 'score:', scores, 'action:', action);


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
        <canvas
            ref={canvasRef}
            width={600}
            height={400}
        />
    );
}