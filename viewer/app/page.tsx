"use client";

import dynamic from "next/dynamic";

const ReplayViewer = dynamic(() => import("@/components/ReplayViewer"), {
  ssr: false,
  loading: () => (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        height: "100vh",
        color: "#a8c4ff",
      }}
    >
      Loading 3D viewer…
    </div>
  ),
});

export default function Page() {
  return <ReplayViewer />;
}
