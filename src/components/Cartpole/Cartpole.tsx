import { useEffect, useRef, useState } from 'react';
import * as ort from 'onnxruntime-web';

ort.env.wasm.numThreads = 1;

type CartPoleProps = { modelPath?: string };

export default function CartPole({ modelPath = '/cartpole/model.onnx' }: CartPoleProps) {
    const canvasRef = useRef<HTMLCanvasElement | null>(null);

    const [actionState, setActionState] = useState<number>(0); // 0 = left, 1 = right
    const [, setFlashKey] = useState<number>(0);       // setter only to retrigger CSS animation
    const [error, setError] = useState<string | null>(null);
    const scoresRef = useRef<number[]>([0, 0]);
    const lastActionRef = useRef<number>(0);

    useEffect(() => {
        let isMounted = true;
        let session: ort.InferenceSession | undefined;
        let inputName: string | undefined;
        let outputName: string | undefined;
        const run = async () => {
            try {
                const canvas = canvasRef.current;
                if (!canvas) return;

                const mod = await import('./Cartpole.js');
                const { CartPoleEnv: EnvCtor, CartPoleRenderer: RendererCtor } = mod as any;
                const env = new EnvCtor();
                const renderer = new RendererCtor(canvas, env);
                // Draw the initial state
                renderer.draw(env.reset());

                // Load the model once
                session = await ort.InferenceSession.create(modelPath);
                // Use the model's actual input/output names
                inputName = (session as any).inputNames?.[0] as string | undefined;
                outputName = (session as any).outputNames?.[0] as string | undefined;

                let input = new ort.Tensor(
                    'float32',
                    Float32Array.from(env.state),
                    [1, env.state.length]
                );

                // Main loop (~20 FPS)
                while (isMounted) {
                    if (!session || !inputName || !outputName) break;

                    // Run inference with the correct key
                    const outputMap = await session.run({ [inputName]: input });

                    // Read the output tensor by its actual name
                    const tensor = (outputMap as any)[outputName];
                    const scores = Array.from(tensor.data as Float32Array);
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

                    const { state: nextState, done } = env.step(action);
                    renderer.draw(nextState);
                    input = new ort.Tensor(
                        'float32',
                        Float32Array.from(nextState),
                        [1, nextState.length]
                    );

                    if (done) env.reset();

                    await new Promise(r => setTimeout(r, 25));
                }
            } catch (err) {
                console.error('CartPole runtime error:', err);
                if (isMounted) setError((err as any)?.message || String(err));
            }
        };

        run();

        return () => {
            isMounted = false;
        };
    }, [modelPath]);

    const raw = scoresRef.current ?? [0, 0];
    const exps = raw.map(x => Math.exp(x));
    const sum = (exps[0] + exps[1]) || 1;
    const leftPct = Math.round((exps[0] / sum) * 100);
    const rightPct = 100 - leftPct;

    return (
        <div style={{ display: "flex", flexDirection: "row", width: 600, height: 400 }}>
            {error && (
                <div style={{ position: 'absolute', top: 16, left: 16, right: 16, padding: 8, background: 'rgba(200,0,0,0.85)', color: '#fff', borderRadius: 6, zIndex: 1000 }}>
                    <strong>CartPole error:</strong> {error}
                </div>
            )}

            <canvas
                ref={canvasRef}
                width={1200}
                height={800}
                style={{ display: "block" }}
            />

            {/* HUD overlay in the bottom-right corner */}
            <div
                className={`cp-hud ${actionState === 1 ? "right" : "left"}`}
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
