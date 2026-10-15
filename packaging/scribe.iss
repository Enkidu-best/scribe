; Установщик Scribe для Windows (Inno Setup 6). Собирается на GitHub Actions после PyInstaller.
; Ставится без прав администратора (в папку пользователя), ярлык в меню «Пуск», удаление - через «Программы».
#ifndef AppVersion
  #define AppVersion "2.4"
#endif

[Setup]
AppId={{8182D8C6-EC7C-45FF-BF18-2021FC678CD5}
AppName=Scribe
AppVersion={#AppVersion}
AppVerName=Scribe {#AppVersion}
AppPublisher=Scribe
DefaultDirName={autopf}\Scribe
DefaultGroupName=Scribe
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir=..\dist
OutputBaseFilename=Scribe-Setup-{#AppVersion}
SetupIconFile=scribe.ico
UninstallDisplayIcon={app}\Scribe.exe
Compression=lzma2/normal
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "ru"; MessagesFile: "compiler:Languages\Russian.isl"
Name: "en"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "..\dist\Scribe\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Scribe"; Filename: "{app}\Scribe.exe"
Name: "{group}\Удалить Scribe"; Filename: "{uninstallexe}"
Name: "{autodesktop}\Scribe"; Filename: "{app}\Scribe.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\Scribe.exe"; Description: "{cm:LaunchProgram,Scribe}"; Flags: nowait postinstall skipifsilent
