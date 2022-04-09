package com.example.myapplication;

import android.media.MediaPlayer;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.view.View;
import android.widget.ImageButton;
import android.widget.SeekBar;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

/*
This page is used to display previously recorded speeches
 */
public class HistoryActivity extends AppCompatActivity implements PlayAudioInterface{

    RecyclerView recyclerView;
    SeekBar seekBar;
    ImageButton playBtn, previousBtn, nextBtn;
    TextView startTv, endTv;
    RecyclerViewAdapter adapter;
    PlayAudioInterface playAudioInterface;
    ArrayList<File> audioList;
    MediaPlayer mediaPlayer;
    Handler handler;
    Runnable runnable;
    int progress;
    int posit=0;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_history);

        playAudioInterface= this;
        recyclerView= findViewById(R.id.recyclerView);
        seekBar= findViewById(R.id.seekBar);
        playBtn= findViewById(R.id.play_icon);
        previousBtn= findViewById(R.id.previose_icon);
        nextBtn= findViewById(R.id.next_icon);
        startTv= findViewById(R.id.startTv);
        endTv= findViewById(R.id.endTv);
        mediaPlayer= new MediaPlayer();
        handler= new Handler();

        playBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (mediaPlayer.isPlaying()){
                    playBtn.setImageResource(R.drawable.play_icon);
                    mediaPlayer.pause();
                }
                else {
                    playBtn.setImageResource(R.drawable.pause_icon);
                    mediaPlayer.start();
                }
            }
        });

        nextBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                handler.removeCallbacks(runnable);
                if (!(posit <audioList.size()-1)){
                    posit=0;
                }
                else posit++;
                playAudioInterface.playAudio(posit);
            }
        });
        previousBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                handler.removeCallbacks(runnable);
                if (!(posit >0)){
                    posit=0;
                }
                else posit--;
                playAudioInterface.playAudio(posit);
            }
        });

        recyclerView.setLayoutManager(new LinearLayoutManager(this));

        getHistroyFiles();
    }

    private void getHistroyFiles() {
        audioList= new ArrayList<>();

        String mainPath= Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_MUSIC).getPath();

//        File file = new File( mainPath + "/recorded" );
        File file = new File( mainPath + "/Speeches" );
        File list[] = file.listFiles();

        for( int i=0; i< list.length; i++)
        {
//            if(checkExtension( list[i].getName() ))
//            {
                audioList.add( list[i] );
//            }
        }

        adapter= new RecyclerViewAdapter(this, audioList, playAudioInterface);
        recyclerView.setAdapter(adapter);

    }

    //Function of the audio player. Allows the user to play recordings with a media player
    @Override
    public void playAudio(int position) {
        posit= position;
        try {
            if (mediaPlayer.isPlaying()){
                mediaPlayer.stop();
            }
            mediaPlayer= new MediaPlayer();
            mediaPlayer.setDataSource(this, Uri.fromFile(audioList.get(position)));
            mediaPlayer.prepare();
            mediaPlayer.start();
            mediaPlayer.setOnCompletionListener(new MediaPlayer.OnCompletionListener() {
                @Override
                public void onCompletion(MediaPlayer mp) {
                    handler.removeCallbacks(runnable);
                    startTv.setText("00");
                    seekBar.setProgress(0);
                }
            });
            int duration= mediaPlayer.getDuration();
            int minute= (duration/1000)/60;
            int second= duration/1000;
            String dur= String.format("%02d:%02d", minute, second);
            endTv.setText(dur);
            seekBar.setMax(second);
            progress=0;
            runnable= new Runnable() {
                @Override
                public void run() {
                    seekBar.setProgress(progress);
                    startTv.setText(String.format("%02d", progress));
                    progress++;
                    handler.postDelayed(runnable, 1000);
                }
            };
            handler.postDelayed(runnable, 1000);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void onBackPressed() {
        super.onBackPressed();
        if (mediaPlayer.isPlaying())
            mediaPlayer.stop();
    }
}