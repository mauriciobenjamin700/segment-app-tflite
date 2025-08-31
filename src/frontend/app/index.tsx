import * as RN from "react-native";
import Button from "@/components/Button";
import ImagePreview from "@/components/ImagePreview";


export default function Index() {
  return (
    <RN.View
      style={{
        flex: 1,
        justifyContent: "center",
        alignItems: "center",
      }}
    >
      <ImagePreview />
      <Button title="Buscar Imagem" onPress={() => alert("Pressed!")} />
      <Button title="Iniciar Segmentação" onPress={() => alert("Pressed!")} />
      
    </RN.View>
  );
}
