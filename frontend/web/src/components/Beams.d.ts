import type { FC } from "react";

/** React Bits "Beams" (three.js / react-three-fiber), installed via shadcn. */
declare const Beams: FC<{
  beamWidth?: number;
  beamHeight?: number;
  beamNumber?: number;
  lightColor?: string;
  beamColor?: string;
  backgroundColor?: string;
  speed?: number;
  noiseIntensity?: number;
  scale?: number;
  rotation?: number;
  lightMode?: boolean;
}>;
export default Beams;
