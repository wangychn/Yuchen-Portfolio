import { useEffect, useRef, useState } from 'react';
import * as ort from 'onnxruntime-web';

ort.env.wasm.numThreads = 1;

type CartPoleProps = { modelPath?: string };

export default function CartPole({ modelPath = '/cartpole/model.onnx' }: CartPoleProps) {
    const canvasRef = useRef<HTMLCanvasElement | null>(null);

    const [actionState, setActionState] = useState<number>(0); // 0 = left, 1 = right
    const [, setFlashKey] = useState<number>(0);       // setter only to retrigger CSS animation
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

                const mod = await import('./cartpole_sim.ts');
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

    const displayW = 460;
    const displayH = 160;
    // Render at higher internal resolution, but display at smaller CSS size.
    const renderScale = Math.max(2, Math.round(window.devicePixelRatio || 1));
    // const renderScale = 15;

    return (
        <div
            style={{
                // display: "flex",
                // flexDirection: "row",
                // position: "relative",
                width: displayW,
                height: displayH
            }}
        >
            <canvas
                ref={canvasRef}
                width={displayW * renderScale}
                height={displayH * renderScale}
                style={{ display: "block", width: displayW, height: displayH }}
            />

            {/* HUD overlay in the bottom-right corner */}
            <div
                className={`cp-hud ${actionState === 1 ? "right" : "left"}`}
                style={{
                    position: "absolute",
                    right: 8,
                    top: 200,
                    borderRadius: 10,
                    fontFamily: "var(--font-body, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif)",
                    fontSize: 12,
                    background: "rgba(255, 255, 255, 0.67)",
                    color: "#3b3b3bff",
                    userSelect: "none",
                    pointerEvents: "none",
                    minWidth: 160,
                    maxHeight: 90,
                    boxShadow: "0 2px 8px rgba(0,0,0,0.2)",
                    padding: 8,
                    marginTop: 20,
                }}
            >
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
                    <strong>Model Output</strong>
                    <span style={{
                        padding: "1px 6px",
                        borderRadius: 6,
                        background: actionState === 1 ? "rgba(76, 175, 79, 0.75)" : "rgba(33, 149, 243, 0.63)"
                    }}>
                        {actionState === 1 ? "RIGHT" : "LEFT"}
                    </span>
                </div>

                {/* Left Q Val Bar */}
                <div style={{ marginBottom: 3 }}>
                    <div style={{ display: "flex", justifyContent: "space-between" }}>
                        <span>Left Q Val</span>
                        <span>{scoresRef.current?.[0]?.toFixed?.(2) ?? "-"}</span>
                    </div>
                    <div style={{ height: 4, background: "rgba(255,255,255,0.15)", borderRadius: 4, overflow: "hidden" }}>
                        <div style={{
                            height: "100%",
                            width: `${leftPct}%`,
                            background: "#64b9ffff",
                            transition: "width 120ms linear"
                        }} />
                    </div>
                </div>

                {/* Left Q Val Bar */}
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                    <span>Right Q Val</span>
                    <span>{scoresRef.current?.[1]?.toFixed?.(2) ?? "-"}</span>
                </div>
                <div style={{ height: 4, background: "rgba(255,255,255,0.15)", borderRadius: 4, overflow: "hidden" }}>
                    <div style={{
                        height: "100%",
                        width: `${rightPct}%`,
                        background: "#64ce68ff",
                        transition: "width 120ms linear"
                    }} />
                </div>
            </div>
        </div>
    );
}
